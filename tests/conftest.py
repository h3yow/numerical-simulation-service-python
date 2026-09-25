import time

import pytest
from fastapi.testclient import TestClient

from app.models import TERMINAL_STATES
from app.tasks import TaskManager
from main import create_app


@pytest.fixture()
def manager():
    m = TaskManager(max_concurrent=2, max_queue=4)
    yield m
    m.shutdown()


@pytest.fixture()
def client(manager):
    with TestClient(create_app(manager)) as c:
        yield c


def wait_terminal(client, task_id, timeout=30.0):
    """轮询直到任务进入终态，返回最终状态视图。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        view = client.get(f"/simulations/{task_id}").json()
        if view["status"] in {s.value for s in TERMINAL_STATES}:
            return view
        time.sleep(0.02)
    raise TimeoutError(f"task {task_id} 未在 {timeout}s 内进入终态")
