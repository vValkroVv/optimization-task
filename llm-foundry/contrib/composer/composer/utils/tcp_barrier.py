import socket
import time
from contextlib import closing
import os
import psutil

from composer.utils import dist


def _get_ip_from_iface(iface: str | None) -> str | None:
    """Return IPv4 of given network interface or None if not found."""
    if not iface:
        return None
    addrs = psutil.net_if_addrs().get(iface, [])
    for addr in addrs:
        if addr.family == socket.AF_INET:
            return addr.address
    return None

def _get_default_route_ip() -> str | None:
    """Return default IP address (probably of eth0), or None on failure."""
    try:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return None

def _resolve_ip_address() -> str:
    """
    Resolve IP in priority:

      1. TCP_BARRIER_HOST
      2. TCP_BARRIER_IFACE / IB_INTERFACE
      3. Default IP address
      4. SLURM_LAUNCH_NODE_IPADDR
      5. 127.0.0.1

    """
    resolvers = [
        lambda: os.environ.get("TCP_BARRIER_HOST"),
        lambda: _get_ip_from_iface(os.environ.get("TCP_BARRIER_IFACE") or os.environ.get("IB_INTERFACE")),
        _get_default_route_ip,
        lambda: os.environ.get("SLURM_LAUNCH_NODE_IPADDR"),
    ]

    for resolver in resolvers:
        ip = resolver()
        if ip:
            return ip

    return "127.0.0.1"

class TCPBarrier:
    def __init__(self, parties: int, rank: int, host: str, port: int):
        assert parties > 0
        self._parties = parties
        self._rank = rank
        self._host = host
        self._port = port
        self._server = self._get_server_socket() if rank == 0 else None

    def _get_server_socket(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        original_port = self._port
        max_retries = 10

        for attempt in range(max_retries):
            try:
                sock.bind((self._host, self._port))
                self._port = sock.getsockname()[1]
                sock.listen(self._parties)
                return sock

            except OSError as e:
                sock.close()

                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Failed to bind TCP barrier socket after {max_retries} attempts. "
                        f"Requested port: {original_port}, Last error: {e}. "
                        f"Try: (1) Set TCP_BARRIER_PORT=0 for auto port selection, "
                        f"or (2) Kill process using port: lsof -ti:{original_port} | xargs kill -9"
                    ) from e

                if self._port != 0:
                    self._port = 0

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                time.sleep(0.1 * (attempt + 1))
        raise RuntimeError("Unexpected: exited retry loop without returning socket")

    def _server_wait(self) -> None:
        joined_parties: list[socket.socket] = list()
        while len(joined_parties) < self._parties - 1:
            client_socket, _ = self._server.accept()
            joined_parties.append(client_socket)
        for sock in joined_parties:
            message = "run"
            sock.sendall(message.encode("utf-8"))
            sock.close()

    def _connect_with_backoff(
            self,
            start_timeout: float = 0.5,
            backoff_alpha: float = 2.0,
            max_timeout: float = 4.0,
    ) -> socket.socket:
        timeout = start_timeout
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                sock.connect((self._host, self._port))
                return sock
            except ConnectionRefusedError:
                time.sleep(timeout)
                timeout = min(timeout * backoff_alpha, max_timeout)

    def _client_wait(self):
        with closing(self._connect_with_backoff()) as sock:
            sock.recv(1024)

    def wait(self):
        if self._rank == 0:
            self._server_wait()
        else:
            self._client_wait()

    def __del__(self):
        if self._server is not None:
            try:
                self._server.close()
            except Exception:
                pass

def get_tcp_barrier(parties: int) -> TCPBarrier:
    rank = dist.get_global_rank()

    _master_ip_addr = [_resolve_ip_address()] if rank == 0 else [None]
    dist.broadcast_object_list(_master_ip_addr)
    master_ip_addr = _master_ip_addr[0]

    port = int(os.environ.get("TCP_BARRIER_PORT", "0"))

    if rank == 0:
        barrier = TCPBarrier(
            parties=parties,
            rank=rank,
            host=master_ip_addr,
            port=port,
        )
        actual_port = [barrier._port]
    else:
        actual_port = [None]

    dist.broadcast_object_list(actual_port)

    if rank != 0:
        barrier = TCPBarrier(
            parties=parties,
            rank=rank,
            host=master_ip_addr,
            port=actual_port[0],
        )

    return barrier
