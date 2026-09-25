"""场景三：相同随机种子与参数必须得到可复现结果；状态与结果在各查询入口一致。"""
from tests.conftest import wait_terminal

BASE = {
    "nx": 128,
    "dt": 1e-4,
    "num_steps": 500,
    "diffusion": 0.05,
    "advection": 0.3,
    "boundary": "periodic",
    "initial": "random",
    "noise_amplitude": 0.2,
}


def _run(client, payload):
    r = client.post("/simulations", json=payload)
    assert r.status_code == 201, r.json()
    tid = r.json()["task_id"]
    view = wait_terminal(client, tid)
    res = client.get(f"/simulations/{tid}/result").json()
    return view, res


def test_same_seed_reproducible(client):
    payload = {**BASE, "seed": 123}
    view1, res1 = _run(client, payload)
    view2, res2 = _run(client, payload)
    print(f"[reproduce] seed={payload['seed']}, nx={payload['nx']}, "
          f"dt={payload['dt']}, steps={payload['num_steps']}")
    print(f"[reproduce] run1: status={res1['status']}, checksum={res1['checksum']}")
    print(f"[reproduce] run2: status={res2['status']}, checksum={res2['checksum']}")
    assert view1["status"] == view2["status"] == "succeeded"
    assert res1["checksum"] == res2["checksum"]
    assert res1["final_state"] == res2["final_state"]


def test_different_seed_differs(client):
    _, res1 = _run(client, {**BASE, "seed": 1})
    _, res2 = _run(client, {**BASE, "seed": 2})
    print(f"[reproduce] seed=1 checksum={res1['checksum']} vs "
          f"seed=2 checksum={res2['checksum']}（判定依据：不同种子应产生不同末态）")
    assert res1["final_state"] != res2["final_state"]


def test_status_and_result_consistent_across_endpoints(client):
    """同一任务在状态查询与结果查询两个入口下的状态/错误信息必须一致。"""
    payload = {**BASE, "seed": 99}
    tid = client.post("/simulations", json=payload).json()["task_id"]
    view = wait_terminal(client, tid)
    res = client.get(f"/simulations/{tid}/result").json()
    print(f"[consistency] 状态入口: status={view['status']}, "
          f"reason={view['error_reason']}; 结果入口: status={res['status']}, "
          f"reason={res['error_reason']}")
    assert view["status"] == res["status"]
    assert view["error_reason"] == res["error_reason"]
    assert view["progress"] == 1.0
    assert res["nx"] == payload["nx"]
    assert len(res["final_state"]) == payload["nx"]


def test_result_not_ready_returns_409(client):
    payload = {**BASE, "seed": 5, "num_steps": 100_000, "nx": 200}
    tid = client.post("/simulations", json=payload).json()["task_id"]
    r = client.get(f"/simulations/{tid}/result")
    print(f"[consistency] 未完成时取结果 -> http={r.status_code}（判定依据：应为 409）")
    assert r.status_code == 409
    client.post(f"/simulations/{tid}/cancel")
    wait_terminal(client, tid)
