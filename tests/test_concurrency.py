"""场景一：并发提交、排队、取消与重复取消下的状态一致性。"""
import time
from concurrent.futures import ThreadPoolExecutor

from tests.conftest import wait_terminal

# 足够长的任务，保证可以观察到 running/queued 状态并来得及取消
LONG = {"nx": 200, "dt": 1e-4, "num_steps": 100_000, "diffusion": 0.1,
        "seed": 11, "noise_amplitude": 0.1}
SHORT = {"nx": 64, "dt": 1e-4, "num_steps": 300, "diffusion": 0.1,
         "seed": 11, "noise_amplitude": 0.1}


def _submit(client, payload):
    r = client.post("/simulations", json=payload)
    assert r.status_code == 201, r.json()
    return r.json()["task_id"]


def test_bounded_concurrency_and_queueing(client, manager):
    """max_concurrent=2：任意时刻运行中的任务不超过 2，其余排队等待并最终完成。"""
    tids = [_submit(client, LONG) for _ in range(4)]
    for tid in tids:
        view = wait_terminal(client, tid)
        assert view["status"] == "succeeded", view
    observed = manager.max_running_observed
    print(f"[concurrency] 提交 {len(tids)} 个长任务, max_concurrent=2, "
          f"管理器观测到的最大并发运行数={observed}（判定依据：<=2）")
    assert 1 <= observed <= 2
    print(f"[concurrency] 全部 {len(tids)} 个任务最终均 succeeded，排队任务未被丢弃")


def test_queue_full_rejected(client):
    """max_concurrent=2 + max_queue=4：第 7 个任务必须被拒绝且原因为 resource_limit。"""
    tids = [_submit(client, LONG) for _ in range(6)]
    r = client.post("/simulations", json=LONG)
    print(f"[concurrency] 队列满时第 7 个提交 -> http={r.status_code}, "
          f"body={r.json()}（判定依据：429 + resource_limit）")
    assert r.status_code == 429
    assert r.json()["error_reason"] == "resource_limit"
    for tid in tids:
        client.post(f"/simulations/{tid}/cancel")
    for tid in tids:
        wait_terminal(client, tid)


def test_cancel_queued_task(client):
    """排队中的任务被取消后不得再被执行。"""
    blockers = [_submit(client, LONG) for _ in range(2)]  # 占满 worker
    queued = _submit(client, LONG)
    time.sleep(0.1)
    s = client.get(f"/simulations/{queued}").json()["status"]
    print(f"[cancel] 目标任务当前状态={s}（预期 queued）")
    assert s == "queued"
    view = client.post(f"/simulations/{queued}/cancel").json()
    print(f"[cancel] 取消排队任务 -> status={view['status']}, "
          f"reason={view['error_reason']}")
    assert view["status"] == "cancelled"
    assert view["error_reason"] == "cancelled"
    final = wait_terminal(client, queued)
    assert final["status"] == "cancelled"  # 不会重新被调度执行
    res = client.get(f"/simulations/{queued}/result").json()
    assert res["status"] == "cancelled" and res["final_state"] is None
    for tid in blockers:
        client.post(f"/simulations/{tid}/cancel")
        wait_terminal(client, tid)


def test_cancel_running_task(client):
    tid = _submit(client, LONG)
    deadline = time.time() + 10
    while client.get(f"/simulations/{tid}").json()["status"] != "running":
        assert time.time() < deadline
        time.sleep(0.01)
    view = client.post(f"/simulations/{tid}/cancel").json()
    print(f"[cancel] 运行中取消 -> 即时状态={view['status']}")
    final = wait_terminal(client, tid)
    res = client.get(f"/simulations/{tid}/result").json()
    print(f"[cancel] 终态={final['status']}, reason={final['error_reason']}, "
          f"progress={final['progress']:.3f}（判定依据：cancelled 且无结果）")
    assert final["status"] == "cancelled"
    assert final["error_reason"] == "cancelled"
    assert res["status"] == "cancelled"
    assert res["final_state"] is None


def test_double_cancel_idempotent(client):
    tid = _submit(client, LONG)
    v1 = client.post(f"/simulations/{tid}/cancel").json()
    v2 = client.post(f"/simulations/{tid}/cancel").json()
    final = wait_terminal(client, tid)
    print(f"[cancel] 重复取消: 第一次={v1['status']}, 第二次={v2['status']}, "
          f"终态={final['status']}（判定依据：幂等，终态唯一）")
    assert final["status"] == "cancelled"
    assert v2["status"] in ("cancelled", "running")  # 不报错、不变更语义
    # 取消后再取消 / 查询，状态不回退不错乱
    assert client.post(f"/simulations/{tid}/cancel").json()["status"] == "cancelled"


def test_cancel_terminal_task_is_noop(client):
    tid = _submit(client, SHORT)
    final = wait_terminal(client, tid)
    assert final["status"] == "succeeded"
    view = client.post(f"/simulations/{tid}/cancel").json()
    res = client.get(f"/simulations/{tid}/result").json()
    print(f"[cancel] 已完成任务收到取消 -> status={view['status']} "
          f"（判定依据：保持 succeeded，结果仍可取）")
    assert view["status"] == "succeeded"
    assert res["status"] == "succeeded"
    assert res["final_state"] is not None


def test_concurrent_submit_and_cancel_from_threads(client):
    """多线程并发提交 + 并发取消：所有任务到达唯一终态，查询入口结果一致。"""
    def submit_one(i):
        r = client.post("/simulations", json={**SHORT, "seed": i})
        return r.status_code, r.json()

    with ThreadPoolExecutor(max_workers=8) as pool:
        submitted = list(pool.map(submit_one, range(20)))
    tids = [b["task_id"] for code, b in submitted if code == 201]
    rejected = [b for code, b in submitted if code != 201]
    print(f"[concurrency] 20 个并发提交: 接受 {len(tids)}, 拒绝 {len(rejected)}")
    assert all(b["error_reason"] == "resource_limit" for b in rejected)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda t: client.post(f"/simulations/{t}/cancel"), tids))

    for tid in tids:
        view = wait_terminal(client, tid)
        res = client.get(f"/simulations/{tid}/result").json()
        assert view["status"] in ("succeeded", "cancelled")
        assert res["status"] == view["status"]  # 两个查询入口一致
        if view["status"] == "cancelled":
            assert view["error_reason"] == "cancelled"
            assert res["final_state"] is None
        else:
            assert res["final_state"] is not None
    print(f"[concurrency] {len(tids)} 个任务全部到达唯一终态，两入口状态一致")
