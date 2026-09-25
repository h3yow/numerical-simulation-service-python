"""Tests for the numerical simulation task service.

Covers three required scenarios:
  1. concurrent submission and cancellation;
  2. invalid / numerically unstable / over-limit parameters;
  3. reproducibility for identical (parameters, seed).

Each test logs the key input parameters and the judgment basis used.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from main import app, task_manager
from simulation_service.errors import ErrorReason
from simulation_service.models import TaskStatus

logger = logging.getLogger("test.simulation_service")

client = TestClient(app)

TERMINAL = {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED}

BASE_PARAMS = {
    "grid_size": 101,
    "domain_length": 1.0,
    "dt": 1e-4,
    "num_steps": 50,
    "advection_velocity": 0.5,
    "diffusion_coefficient": 0.01,
    "boundary_condition": "periodic",
    "initial_condition": "gaussian",
    "noise_amplitude": 0.0,
    "seed": None,
}


def submit(params: dict):
    return client.post("/simulations", json=params)


def wait_terminal(task_id: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/simulations/{task_id}").json()
        if TaskStatus(body["status"]) in TERMINAL:
            return body
        time.sleep(0.02)
    raise AssertionError(f"task {task_id} did not reach a terminal state")


# ---------------------------------------------------------------------------
# Scenario 2: invalid / unstable / over-limit parameters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("override", "expected_reason"),
    [
        ({"dt": -1.0}, ErrorReason.INVALID_PARAMETERS),
        ({"diffusion_coefficient": -0.1}, ErrorReason.INVALID_PARAMETERS),
        ({"noise_amplitude": 0.5}, ErrorReason.INVALID_PARAMETERS),  # missing seed
        ({"dt": 1.0}, ErrorReason.NUMERICALLY_UNSTABLE),  # diffusion number 100 > 0.5
        ({"advection_velocity": 1e6}, ErrorReason.NUMERICALLY_UNSTABLE),  # CFL >> 1
        ({"grid_size": 100_001}, ErrorReason.RESOURCE_LIMIT_EXCEEDED),
        ({"num_steps": 200_000}, ErrorReason.RESOURCE_LIMIT_EXCEEDED),
    ],
)
def test_rejected_parameters_have_distinguishable_reasons(override, expected_reason):
    params = {**BASE_PARAMS, **override}
    resp = submit(params)
    body = resp.json()
    logger.info(
        "rejection check | override=%s -> http=%s reason=%s detail=%s",
        override, resp.status_code, body.get("error_reason"), body.get("error_detail"),
    )
    assert resp.status_code == 400
    assert body["status"] == TaskStatus.FAILED
    assert body["error_reason"] == expected_reason
    # The rejected task never computes: progress stays 0 and no result exists.
    assert body["progress"] == 0.0
    result_resp = client.get(f"/simulations/{body['task_id']}/result")
    assert result_resp.status_code == 409


def test_reasons_are_mutually_distinguishable():
    cases = [
        ({"dt": -1.0}, ErrorReason.INVALID_PARAMETERS),
        ({"dt": 1.0}, ErrorReason.NUMERICALLY_UNSTABLE),
        ({"grid_size": 100_001}, ErrorReason.RESOURCE_LIMIT_EXCEEDED),
    ]
    reasons = set()
    for override, expected in cases:
        body = submit({**BASE_PARAMS, **override}).json()
        logger.info("distinguish | override=%s -> reason=%s", override, body["error_reason"])
        assert body["error_reason"] == expected
        reasons.add(body["error_reason"])
    assert len(reasons) == len(cases), "failure reasons must be distinguishable"


# ---------------------------------------------------------------------------
# Scenario 3: same seed + parameters -> reproducible results
# ---------------------------------------------------------------------------

def test_same_seed_reproducible_and_different_seed_differs():
    params = {
        **BASE_PARAMS,
        "initial_condition": "random",
        "noise_amplitude": 0.2,
        "seed": 1234,
    }
    results = []
    for i in range(2):
        resp = submit(params)
        assert resp.status_code == 201
        task_id = resp.json()["task_id"]
        final = wait_terminal(task_id)
        assert final["status"] == TaskStatus.SUCCEEDED
        result = client.get(f"/simulations/{task_id}/result").json()
        results.append(result["final_state"])
        logger.info(
            "repro run %d | seed=%s steps=%s checksum=%.12f",
            i, params["seed"], result["steps_completed"], sum(result["final_state"]),
        )
    assert results[0] == results[1], "identical seed+params must reproduce identical results"

    other = submit({**params, "seed": 4321}).json()
    wait_terminal(other["task_id"])
    other_state = client.get(f"/simulations/{other['task_id']}/result").json()["final_state"]
    logger.info(
        "repro contrast | seed=1234 checksum=%.12f vs seed=4321 checksum=%.12f",
        sum(results[0]), sum(other_state),
    )
    assert results[0] != other_state, "different seeds should give different random draws"


def test_status_consistent_across_entry_points():
    resp = submit(BASE_PARAMS)
    task_id = resp.json()["task_id"]
    final = wait_terminal(task_id)
    again = client.get(f"/simulations/{task_id}").json()
    logger.info("consistency | status=%s progress=%s", again["status"], again["progress"])
    assert again == final, "repeated queries of a terminal task must agree"
    assert final["status"] == TaskStatus.SUCCEEDED
    assert final["progress"] == 1.0


# ---------------------------------------------------------------------------
# Scenario 1: concurrent submission and cancellation
# ---------------------------------------------------------------------------

def test_concurrent_submit_and_cancel():
    slow = {**BASE_PARAMS, "grid_size": 501, "num_steps": 20000}
    n_tasks = 8
    logger.info(
        "concurrency | submitting %d tasks grid=%d steps=%d (max_concurrent=2)",
        n_tasks, slow["grid_size"], slow["num_steps"],
    )
    with ThreadPoolExecutor(max_workers=n_tasks) as pool:
        snapshots = list(pool.map(lambda _: submit(slow).json(), range(n_tasks)))
    assert all(s["status"] in (TaskStatus.QUEUED, TaskStatus.RUNNING) for s in snapshots)
    task_ids = [s["task_id"] for s in snapshots]
    assert len(set(task_ids)) == n_tasks

    # Cancel half of them concurrently, including double-cancel of one task.
    to_cancel = task_ids[: n_tasks // 2]
    with ThreadPoolExecutor(max_workers=len(to_cancel) + 1) as pool:
        cancel_results = list(pool.map(lambda tid: client.post(f"/simulations/{tid}/cancel").json(), to_cancel))
        double = client.post(f"/simulations/{to_cancel[0]}/cancel").json()
    for body in cancel_results:
        logger.info(
            "cancel | task=%s status=%s reason=%s",
            body["task_id"][:8], body["status"], body["error_reason"],
        )
        assert TaskStatus(body["status"]) in TERMINAL
    assert double["status"] == cancel_results[0]["status"], "repeated cancel must be idempotent"

    finals = [wait_terminal(tid) for tid in task_ids]
    cancelled = [f for f in finals if f["status"] == TaskStatus.CANCELLED]
    succeeded = [f for f in finals if f["status"] == TaskStatus.SUCCEEDED]
    logger.info(
        "concurrency outcome | cancelled=%d succeeded=%d", len(cancelled), len(succeeded)
    )
    assert len(cancelled) >= 1, "at least one task should have been cancelled"
    for f in cancelled:
        assert f["error_reason"] == ErrorReason.CANCELLED
        assert client.get(f"/simulations/{f['task_id']}/result").status_code == 409
    for f in succeeded:
        assert f["progress"] == 1.0
        result = client.get(f"/simulations/{f['task_id']}/result")
        assert result.status_code == 200
        state = result.json()["final_state"]
        assert len(state) == slow["grid_size"]
        assert all(abs(v) < 1e6 for v in state), "succeeded results must be finite and bounded"

    # Cancelling an already-succeeded task must not corrupt its state.
    if succeeded:
        tid = succeeded[0]["task_id"]
        after = client.post(f"/simulations/{tid}/cancel").json()
        logger.info("cancel-on-terminal | status=%s (must stay succeeded)", after["status"])
        assert after["status"] == TaskStatus.SUCCEEDED
        assert client.get(f"/simulations/{tid}/result").status_code == 200


def test_queueing_instead_of_overload():
    slow = {**BASE_PARAMS, "grid_size": 501, "num_steps": 20000}
    snapshots = [submit(slow).json() for _ in range(4)]
    statuses = [client.get(f"/simulations/{s['task_id']}").json()["status"] for s in snapshots]
    logger.info("queueing | concurrent statuses observed: %s", statuses)
    assert TaskStatus.QUEUED in [TaskStatus(s) for s in statuses], (
        "with max_concurrent=2, extra tasks must queue instead of failing"
    )
    for s in snapshots:
        final = wait_terminal(s["task_id"])
        assert final["status"] == TaskStatus.SUCCEEDED
    task_manager.shutdown  # noqa: B018 - manager shared via app fixture
