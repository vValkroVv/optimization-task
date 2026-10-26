import pytest
import multiprocessing as mp
from multiprocessing import Pool
from contextlib import contextmanager
from composer.utils.tcp_barrier import TCPBarrier
import signal

import socket


def _get_free_port() -> int:
    sock = socket.socket()
    sock.bind(('', 0))
    return sock.getsockname()[1]

    
@contextmanager
def timeout(seconds: int = 1):
    def _timeout_handler(signum: int, frame) -> None:
        raise TimeoutError(f"Execution exceeds timeout limit of {seconds} seconds")

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(seconds)
    yield
    signal.alarm(0)
    
    
def _runner(
    parties: int,
    rounds: int,
    rank: int,
    host: str,
    port: int,
) -> None:
    barrier = TCPBarrier(parties, rank, host, port)
    for _ in range(rounds):
        barrier.wait()


@pytest.mark.parametrize("parties,rounds", [(1, 10), (2, 10), (128, 2)])
def test_makes_progress(parties: int, rounds: int):
    port = _get_free_port()
    with Pool(parties) as pool:
        args = [(parties, rounds, rank, "localhost", port) for rank in range(parties)]
        pool.starmap(_runner, args)


def _queue_runner(queue: mp.Queue, parties: int, rank: int, host: str, port: int) -> None:
    barrier = TCPBarrier(parties, rank, host, port)
    queue.put(None, block=False)
    barrier.wait()


def test_server_late_start():
    port = _get_free_port()
    notify_start_waiting = mp.Queue()    

    # start rank 0 (server) barrier after rank 1 (client) barrier
    rank1 = mp.Process(target=_queue_runner, args=(notify_start_waiting, 2, 1, "localhost", port))
    rank1.start()

    notify_start_waiting.get()

    rank0 = mp.Process(target=_queue_runner, args=(notify_start_waiting, 2, 0, "localhost", port))
    rank0.start()

    for task in [rank0, rank1]:
        task.join()


@pytest.mark.parametrize("parties", [2, 4])
def test_deadlock(parties: int):
    port =_get_free_port()
    with pytest.raises(TimeoutError), timeout(seconds=5), Pool(parties - 1) as pool:
        args = [(parties, 1, rank, "localhost", port) for rank in range(parties - 1)]
        pool.starmap(_runner, args)
