import threading
from threading import Lock
from typing import Optional

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed

from .logger import Logger
from .trace import EventsStream

ONE_WEEK_OF_SECONDS = 7 * 24 * 3600


class BarrierBody(BaseModel):
    node_name: str
    stage: str
    expected: int
    node_rank: int


class BarrierResponse(BaseModel):
    released: bool
    arrived: int
    expected: int
    stage: str


class FastApiOrkestrationManager:
    def __init__(self):
        self._lock = Lock()
        self._barriers: dict[str, dict] = {}
        self._ops: list[dict] = []
        self._app: Optional[FastAPI] = None
        self._app_thread: Optional[threading.Thread] = None
        self._app_server: Optional[uvicorn.Server] = None

        # cluster topology
        self.global_nnodes: int = None
        self.global_node_rank: int = None
        self.node_name: str = None
        self.global_dist_init_host: str = None
        self.orkestration_port: int = None

        # node role
        self.is_master = False
        self.is_worker = False

        # registered nodes (rank -> host)
        self._registered_nodes: dict[int, str] = {}

    def _start_app(self, orkestration_port: int):
        if self.is_worker:
            return
        config = uvicorn.Config(
            self._app, host="0.0.0.0", port=orkestration_port, log_level="error"
        )
        self._app_server = uvicorn.Server(config)
        t = threading.Thread(target=self._app_server.run, daemon=True)
        t.start()
        self._app_thread = t

    def _set_master_or_worker(self, node_name: str):
        if node_name == "mpimaster-0":
            self.is_master = True
        else:
            self.is_worker = True

    def _assert_setup(self):
        if self.is_master:
            assert self._app is not None, "App is not set"
            assert self._app_thread is not None, "App thread is not set"

        # cluster required
        assert self.node_name is not None, "Node name is not set"
        assert self.global_node_rank is not None, "Global node rank is not set"
        assert self.global_nnodes is not None, "Nnodes is not set"
        assert self.global_node_rank < self.global_nnodes, (
            "Global node rank cant be greater than nnodes"
        )
        assert self.global_dist_init_host is not None, (
            "Global dist init host is not set"
        )
        assert self.orkestration_port is not None, "Orkestration port is not set"

        assert self.global_nnodes > 0, "Nnodes must be positive"

    def setup(
        self,
        node_name: str,
        orkestration_port: int,
        global_nnodes: int,
        global_node_rank: int,
        global_dist_init_host: str,
    ):
        self._set_master_or_worker(node_name)
        self._build_app()
        self._start_app(orkestration_port)
        self.global_node_rank = global_node_rank
        self.global_nnodes = global_nnodes
        self.orkestration_port = orkestration_port
        self.node_name = node_name
        self.global_dist_init_host = global_dist_init_host
        self._assert_setup()
        Logger.info("Orkestration initialized successfully")

    @retry(
        stop=stop_after_attempt(ONE_WEEK_OF_SECONDS), wait=wait_fixed(1), reraise=True
    )
    def wait_barrier(
        self,
        stage: str,
        expected: int | None = None,
    ) -> bool:
        expected_nodes = expected or self.global_nnodes
        resp = requests.post(
            f"http://{self.global_dist_init_host}:{self.orkestration_port}/barrier",
            json={
                "node_name": self.node_name,
                "stage": stage,
                "expected": expected_nodes,
                "node_rank": self.global_node_rank,
            },
            timeout=5,
        )
        resp.raise_for_status()
        data = BarrierResponse(**resp.json())

        if data.released:
            Logger.info(f"Barrier '{stage}' released: {data.arrived}/{data.expected}")
            return True
        raise RuntimeError("Barrier not released yet")

    def barrier_released(self, stage: str) -> bool:
        with self._lock:
            barrier = self._barriers.get(stage)
            return barrier is not None and barrier["released"]

    def publish_op(self, op: dict) -> int:
        with self._lock:
            op_index = len(self._ops)
            self._ops.append(op)
            return op_index

    @retry(
        stop=stop_after_attempt(ONE_WEEK_OF_SECONDS), wait=wait_fixed(1), reraise=True
    )
    def fetch_op(self, index: int) -> dict:
        op_url = (
            f"http://{self.global_dist_init_host}:{self.orkestration_port}/ops/{index}"
        )
        resp = requests.get(op_url, timeout=5)
        resp.raise_for_status()
        return resp.json()

    def _build_app(self):
        if self.is_worker:
            return

        self._app = FastAPI()

        @self._app.post("/barrier")
        @EventsStream.barrier
        def barrier(body: BarrierBody, request: Request) -> BarrierResponse:
            with self._lock:
                barrier = self._barriers.setdefault(
                    body.stage,
                    {"arrived": set(), "expected": body.expected, "released": False},
                )
                barrier["expected"] = max(barrier["expected"], body.expected)
                barrier["arrived"].add(body.node_name)
                if len(barrier["arrived"]) >= barrier["expected"]:
                    barrier["released"] = True

                if body.node_rank not in self._registered_nodes:
                    Logger.info(
                        f"Registered node {body.node_name} at rank {body.node_rank} with host {request.client.host}"
                    )
                self._registered_nodes[body.node_rank] = request.client.host

                return BarrierResponse(
                    released=barrier["released"],
                    arrived=len(barrier["arrived"]),
                    expected=barrier["expected"],
                    stage=body.stage,
                )

        @self._app.get("/nodes")
        def nodes() -> dict[int, str]:
            return dict(self._registered_nodes)

        @self._app.get("/ops/{index}")
        def get_op(index: int) -> dict:
            with self._lock:
                if not 0 <= index < len(self._ops):
                    raise HTTPException(status_code=404)
                return self._ops[index]

        return self._app

    def get_registered_nodes(self) -> dict[int, str]:
        if self.is_master:
            return dict(self._registered_nodes)

        resp = requests.get(
            f"http://{self.global_dist_init_host}:{self.orkestration_port}/nodes",
            timeout=5,
        )
        resp.raise_for_status()
        return {int(k): v for k, v in resp.json().items()}

    def cleanup(self):
        if self._app_server:
            self._app_server.should_exit = True
        if self._app_thread:
            try:
                self._app_thread.join(timeout=5)
            except Exception:
                pass
