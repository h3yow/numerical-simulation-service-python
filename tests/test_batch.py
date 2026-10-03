"""批量扫参能力测试。

覆盖场景（运行加 -s 可看到每个场景的关键日志）：
1. 组合展开（笛卡尔积、去重拒绝、轴结构非法）与结果对号；
2. 组合数 / 总计算量容量上限在提交时被拒；
3. 批次与单任务共用同一套并发/排队，整批取消与单条取消幂等；
4. 非法/不稳定/超资源组合（含 diffusion=0 纯对流边界）在提交时被拦下，原因可区分；
5. 单条失败/取消不影响同批其他条目；
6. 整批重提同种子结果逐点一致（即使上一批有失败/取消）。
"""
import time

import pytest
from fastapi.testclient import TestClient

from app.models import TERMINAL_STATES, TaskStatus
from app.tasks import TaskManager
from app.batch import BatchManager
from main import create_app

# 短任务：快速跑完
SHORT_BASE = {
    "nx": 64, "dt": 1e-4, "num_steps": 300, "diffusion": 0.1,
    "advection": 0.0, "seed": 1, "noise_amplitude": 0.1,
}
# 长任务：足以观察 queued/running 并来得及取消
LONG_BASE = {
    "nx": 200, "dt": 1e-4, "num_steps": 100_000, "diffusion": 0.1,
    "seed": 1, "noise_amplitude": 0.1,
}

BATCH_TERMINAL = {
    "succeeded", "failed", "cancelled", "partial_failed", "partial_cancelled",
}


@pytest.fixture()
def manager():
    m = TaskManager(max_concurrent=2, max_queue=4)
    yield m
    m.shutdown()


@pytest.fixture()
def batch_manager(manager):
    bm = BatchManager(manager)
    yield bm
    bm.shutdown()


@pytest.fixture()
def client(manager, batch_manager):
    with TestClient(create_app(manager, batch_manager)) as c:
        yield c


def submit_batch(client, base, axes):
    r = client.post("/batches", json={"base": base, "axes": axes})
    return r


def wait_batch(client, batch_id, timeout=60.0):
    """轮询直到批次进入终态（所有条目均终态），返回最终批次视图。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        view = client.get(f"/batches/{batch_id}").json()
        if view["status"] in BATCH_TERMINAL and view["progress"] == 1.0:
            return view
        time.sleep(0.02)
    raise TimeoutError(f"batch {batch_id} 未在 {timeout}s 内进入终态")


# ============ 场景 1：组合展开与结果对号 ============

def test_cartesian_expansion_and_result_mapping(client):
    """两条轴 2x2=4 个组合：每个结果都能对上自己的参数。"""
    axes = [
        {"field": "diffusion", "values": [0.0, 0.1]},
        {"field": "seed", "values": [7, 8]},
    ]
    r = submit_batch(client, SHORT_BASE, axes)
    assert r.status_code == 201, r.json()
    view = r.json()
    bid = view["batch_id"]
    print(f"[expand] 提交批次 {bid[:8]}，total={view['total']}，初始 counts={view['counts']}")
    assert view["total"] == 4
    assert view["counts"]["queued"] == 4

    final = wait_batch(client, bid)
    assert final["status"] == "succeeded", final
    assert final["counts"]["succeeded"] == 4

    res = client.get(f"/batches/{bid}/result").json()
    # 参数对号：结果条目按稳定顺序排列，params 与轴取值一致
    expected = [
        (0.0, 7), (0.0, 8), (0.1, 7), (0.1, 8),
    ]
    for item, (d, s) in zip(res["items"], expected):
        assert item["params"]["diffusion"] == d
        assert item["params"]["seed"] == s
        assert item["status"] == "succeeded"
        assert item["final_state"] is not None
        assert len(item["final_state"]) == SHORT_BASE["nx"]
        print(f"[expand] item={item['item_id']} D={d} seed={s} "
              f"task={item['task_id'][:8]} checksum={item['checksum']:.6f} 对号成功")

    # 状态入口与结果入口一致
    assert res["status"] == final["status"]
    assert res["counts"] == final["counts"]


def test_empty_axes_runs_single_combination(client):
    """不指定扫描轴时，批次只含基础参数一条组合。"""
    r = submit_batch(client, SHORT_BASE, [])
    assert r.status_code == 201, r.json()
    assert r.json()["total"] == 1
    final = wait_batch(client, r.json()["batch_id"])
    assert final["status"] == "succeeded"
    print("[expand] 空轴 -> 单组合批次 succeeded")


def test_duplicate_values_rejected_at_submit(client):
    """轴内重复取值：提交时拒绝（400 invalid_params），不受理任何一条。"""
    r = submit_batch(client, SHORT_BASE, [{"field": "seed", "values": [1, 1, 2]}])
    print(f"[expand] 重复取值 -> http={r.status_code} reason={r.json().get('error_reason')} "
          f"detail={r.json().get('detail')}")
    assert r.status_code == 400
    assert r.json()["error_reason"] == "invalid_params"


def test_duplicate_axis_field_and_unknown_field_rejected(client):
    """同一字段两条轴、不存在的扫描字段：提交时 400 拒绝。"""
    r1 = submit_batch(client, SHORT_BASE, [
        {"field": "nx", "values": [32, 64]},
        {"field": "nx", "values": [48]},
    ])
    print(f"[expand] 重复轴字段 -> http={r1.status_code} reason={r1.json().get('error_reason')}")
    assert r1.status_code == 400 and r1.json()["error_reason"] == "invalid_params"

    r2 = submit_batch(client, SHORT_BASE, [{"field": "not_a_field", "values": [1]}])
    print(f"[expand] 非法轴字段 -> http={r2.status_code} reason={r2.json().get('error_reason')}")
    assert r2.status_code == 400 and r2.json()["error_reason"] == "invalid_params"


# ============ 场景 2：组合数 / 总计算量容量上限在提交时被拒 ============

def test_combination_count_cap_rejected(client):
    """组合数超过单批上限：提交时 429 resource_limit，不悄悄少跑。"""
    r = submit_batch(client, SHORT_BASE, [
        {"field": "seed", "values": list(range(300))},  # 300 > 默认 256
    ])
    body = r.json()
    print(f"[cap] 组合数 300 > 256 -> http={r.status_code} reason={body.get('error_reason')} "
          f"detail={body.get('detail')}")
    assert r.status_code == 429
    assert body["error_reason"] == "resource_limit"
    assert "300" in body["detail"] and "256" in body["detail"]


def test_total_work_cap_rejected(client, manager):
    """组合数合法但总计算量超限：提交时 429，明确给出实际值与上限。"""
    bm_small = BatchManager(manager, max_combinations=256, max_total_work=10_000)
    with TestClient(create_app(manager, bm_small)) as c:
        # 每条 nx*num_steps = 64*300 = 19200，两条即 38400 > 10000
        r = submit_batch(c, SHORT_BASE, [{"field": "seed", "values": [1, 2]}])
        body = r.json()
        print(f"[cap] 总工作量 38400 > 10000 -> http={r.status_code} "
              f"reason={body.get('error_reason')} detail={body.get('detail')}")
        assert r.status_code == 429
        assert body["error_reason"] == "resource_limit"
        assert "38400" in body["detail"] and "10000" in body["detail"]


def test_rejected_batch_runs_nothing(client):
    """被拒批次不得留下任何已受理任务（不先收下再失败）。"""
    before = client.get("/batches/00000000").status_code  # 404 基线
    r = submit_batch(client, SHORT_BASE, [{"field": "seed", "values": list(range(300))}])
    assert r.status_code == 429
    assert "batch_id" not in r.json()
    print(f"[cap] 被拒响应中不含 batch_id（未受理），基线探测 http={before}")


# ============ 场景 3：逐组合提交期校验，原因可区分（含 diffusion=0 边界） ============

def test_one_unstable_combo_rejects_whole_batch_with_violations(client):
    """轴上一个稳定、一个不稳定：整批拒绝，且 violations 逐条标明原因。"""
    r = submit_batch(client, SHORT_BASE, [{"field": "dt", "values": [1e-4, 1.0]}])
    body = r.json()
    print(f"[validate] 含不稳定组合 -> http={r.status_code} reason={body['error_reason']} "
          f"violations={len(body['violations'])}")
    assert r.status_code == 422
    assert body["error_reason"] == "unstable"
    assert len(body["violations"]) == 1
    v = body["violations"][0]
    assert v["error_reason"] == "unstable"
    assert v["params"]["dt"] == 1.0
    print(f"[validate] 违规明细 index={v['index']} detail={v['detail']}")


def test_one_resource_limit_combo_rejected(client):
    """轴扫到 nx 超上限的组合：整批 429 resource_limit。"""
    r = submit_batch(client, SHORT_BASE, [{"field": "nx", "values": [64, 20_000]}])
    body = r.json()
    print(f"[validate] 含超资源组合 -> http={r.status_code} reason={body['error_reason']}")
    assert r.status_code == 429
    assert body["error_reason"] == "resource_limit"
    assert body["violations"][0]["params"]["nx"] == 20_000


def test_diffusion_zero_pure_advection_decision(client):
    """扩散系数扫到 0（纯对流）：能否跑由稳定性规则在提交时给出一致结论。

    - cfl<=1 的纯对流组合：受理并成功；
    - 纯对流但 CFL>1：提交时 422 unstable 拦截。
    """
    base = dict(SHORT_BASE, diffusion=0.0, advection=0.5, noise_amplitude=0.0, seed=None)
    r = submit_batch(client, base, [{"field": "diffusion", "values": [0.0]}])
    assert r.status_code == 201, r.json()
    bid = r.json()["batch_id"]
    final = wait_batch(client, bid)
    print(f"[validate] diffusion=0 纯对流(cfl<=1) -> 批次 {final['status']}")
    assert final["status"] == "succeeded"
    # 同批次内 0 扩散结论一致：重复提交结论相同
    r2 = submit_batch(client, base, [{"field": "diffusion", "values": [0.0]}])
    assert r2.status_code == 201
    print(f"[validate] 重复提交纯对流组合同样受理 http={r2.status_code}")

    bad_base = dict(SHORT_BASE, diffusion=0.0, advection=200.0, noise_amplitude=0.0, seed=None)
    r3 = submit_batch(client, bad_base, [{"field": "diffusion", "values": [0.0]}])
    print(f"[validate] diffusion=0 且 CFL>1 -> http={r3.status_code} "
          f"reason={r3.json()['error_reason']}")
    assert r3.status_code == 422
    assert r3.json()["error_reason"] == "unstable"


def test_seed_required_combo_rejected(client):
    """扫参把某组合变成需要随机成分但无 seed（如 initial 扫到 random）：400。"""
    base = dict(SHORT_BASE, seed=None, noise_amplitude=0.0, initial="gaussian")
    r = submit_batch(client, base, [{"field": "initial", "values": ["gaussian", "random"]}])
    body = r.json()
    print(f"[validate] 扫到 random 但无 seed -> http={r.status_code} "
          f"reason={body['error_reason']} violations={len(body['violations'])}")
    assert r.status_code == 400
    assert body["error_reason"] == "invalid_params"


def test_validation_reasons_are_distinguishable(client):
    """三类提交期原因（invalid/unstable/resource）在批量入口仍可区分。"""
    cases = [
        ([{"field": "seed", "values": [1, 1]}], 400, "invalid_params", "重复取值"),
        ([{"field": "dt", "values": [1.0]}], 422, "unstable", "不稳定"),
        ([{"field": "nx", "values": [20_000]}], 429, "resource_limit", "超资源"),
    ]
    seen = set()
    for axes, http, reason, label in cases:
        r = submit_batch(client, SHORT_BASE, axes)
        print(f"[validate] {label}: http={r.status_code} reason={r.json()['error_reason']}")
        assert r.status_code == http and r.json()["error_reason"] == reason
        seen.add(r.json()["error_reason"])
    assert seen == {"invalid_params", "unstable", "resource_limit"}


# ============ 场景 4：共用并发/排队，整批取消与单条取消幂等 ============

def test_batch_shares_concurrency_and_queue(client, manager):
    """8 条长任务，max_concurrent=2/max_queue=4：运行中始终 <=2，其余排队，不绕过上限。"""
    r = submit_batch(client, LONG_BASE, [{"field": "seed", "values": list(range(8))}])
    view = r.json()
    bid = view["batch_id"]
    assert view["total"] == 8
    time.sleep(0.4)
    v = client.get(f"/batches/{bid}").json()
    print(f"[bconcurrency] 8 条长任务：counts={v['counts']}，"
          f"管理器观测最大并发={manager.max_running_observed}")
    assert v["counts"]["running"] <= 2
    assert manager.max_running_observed <= 2
    # 排队中的条目必须真实处于 queued（待派发或在共享队列中）
    assert v["counts"]["queued"] >= 1
    # 清理
    client.post(f"/batches/{bid}/cancel")
    wait_batch(client, bid)


def test_batch_cancel_queued_items_when_queue_full(client):
    """并发/队列被占满时提交大批：不失败；整批取消后排队与待派发条目全部终止。"""
    # 先用普通单任务占满 worker（验证批次与单任务共享容量）
    blockers = [client.post("/simulations", json={**LONG_BASE, "seed": 100 + i}).json()["task_id"]
                for i in range(2)]
    r = submit_batch(client, LONG_BASE, [{"field": "seed", "values": list(range(8))}])
    assert r.status_code == 201, r.json()
    bid = r.json()["batch_id"]
    time.sleep(0.4)
    v = client.get(f"/batches/{bid}").json()
    print(f"[bcancel] 容量被单任务占满时批次仍受理，counts={v['counts']}（有 queued 即排队中）")
    assert v["counts"]["running"] == 0  # 没有空 worker，批次一条都跑不上
    assert v["counts"]["queued"] == 8

    cv = client.post(f"/batches/{bid}/cancel").json()
    print(f"[bcancel] 整批取消即时 counts={cv['counts']}")
    final = wait_batch(client, bid)
    assert final["counts"]["cancelled"] == 8
    assert final["status"] == "cancelled"
    print(f"[bcancel] 终态 {final['status']} counts={final['counts']}")
    # 收尾占满 worker 的单任务
    for t in blockers:
        client.post(f"/simulations/{t}/cancel")


def test_cancel_whole_batch_idempotent(client):
    """整批取消幂等：重复取消、对终态批次再取消，不重复计算、不回退状态。"""
    bid = submit_batch(client, LONG_BASE,
                       [{"field": "seed", "values": list(range(6))}]).json()["batch_id"]
    v1 = client.post(f"/batches/{bid}/cancel").json()
    v2 = client.post(f"/batches/{bid}/cancel").json()
    final = wait_batch(client, bid)
    v3 = client.post(f"/batches/{bid}/cancel").json()  # 终态后再取消
    print(f"[bcancel] 三次取消：即时1={v1['status']} 即时2={v2['status']} "
          f"终态={final['status']} 终态后={v3['status']} counts={v3['counts']}")
    assert final["status"] == v3["status"]
    assert final["counts"] == v3["counts"]
    assert final["counts"]["succeeded"] == 0
    # 状态入口与结果入口一致
    res = client.get(f"/batches/{bid}/result").json()
    assert res["status"] == final["status"]
    assert all(i["final_state"] is None for i in res["items"])


def test_cancel_single_item_isolates_others(client):
    """取消批次中某一条（一条排队、一条运行）不影响其余条目，且单条取消幂等。"""
    bid = submit_batch(client, LONG_BASE,
                       [{"field": "seed", "values": list(range(8))}]).json()["batch_id"]
    time.sleep(0.4)
    v = client.get(f"/batches/{bid}").json()
    queued_item = next(i["item_id"] for i in v["items"] if i["status"] == "queued")
    running_item = next(i["item_id"] for i in v["items"] if i["status"] == "running")

    u1 = client.post(f"/batches/{bid}/items/{queued_item}/cancel").json()
    u2 = client.post(f"/batches/{bid}/items/{queued_item}/cancel").json()  # 幂等
    client.post(f"/batches/{bid}/items/{running_item}/cancel")
    print(f"[icancel] 取消排队条目 {queued_item} 两次状态={u1['status']}/{u2['status']}；"
          f"取消运行条目 {running_item}")

    final = wait_batch(client, bid)
    print(f"[icancel] 终态 {final['status']} counts={final['counts']}")
    assert final["counts"]["cancelled"] == 2
    assert final["counts"]["succeeded"] == 6
    assert final["status"] == "partial_cancelled"

    # 终态后再取消单条：不回退、不重复计算
    again = client.post(f"/batches/{bid}/items/{queued_item}/cancel").json()
    assert again["counts"] == final["counts"]
    # 不存在的条目 -> 404
    assert client.post(f"/batches/{bid}/items/9999/cancel").status_code == 404

    res = client.get(f"/batches/{bid}/result").json()
    by_id = {i["item_id"]: i for i in res["items"]}
    assert by_id[queued_item]["status"] == "cancelled"
    assert by_id[running_item]["status"] == "cancelled"
    others = [i for iid, i in by_id.items() if iid not in (queued_item, running_item)]
    assert all(i["status"] == "succeeded" and i["final_state"] is not None for i in others)


def test_batch_status_reflects_queued_running_finished(client):
    """批次总状态如实反映排队/运行/结束的数量，且进度为终态占比。"""
    bid = submit_batch(client, LONG_BASE,
                       [{"field": "seed", "values": list(range(4))}]).json()["batch_id"]
    time.sleep(0.4)
    v = client.get(f"/batches/{bid}").json()
    assert v["status"] in ("running",)
    total = sum(v["counts"].values())
    assert total == 4
    assert v["counts"]["running"] >= 1
    print(f"[bstatus] 运行中批次 status={v['status']} counts={v['counts']} "
          f"progress={v['progress']}")
    client.post(f"/batches/{bid}/cancel")
    final = wait_batch(client, bid)
    # 计数之和恒等于 total，终态时 progress=1.0
    assert sum(final["counts"].values()) == final["total"] == 4
    assert final["progress"] == 1.0


# ============ 场景 5：单条失败隔离，整批重提同种子逐点复现 ============

@pytest.fixture()
def inject_run_failure(monkeypatch):
    """让特定 seed 的组合在运行时内部失败（模拟运行中数值发散），其余正常。"""
    import app.tasks as tasks_mod

    real = tasks_mod.run_simulation

    def patched(req, **kwargs):
        if req.seed == 3:
            raise RuntimeError("第 256 步出现非有限值，数值发散")
        return real(req, **kwargs)

    monkeypatch.setattr(tasks_mod, "run_simulation", patched)
    return patched


def _run_batch_and_get_result(client, base, seeds):
    bid = submit_batch(client, base, [{"field": "seed", "values": seeds}]).json()["batch_id"]
    final = wait_batch(client, bid)
    res = client.get(f"/batches/{bid}/result").json()
    return final, res


def test_failed_item_isolated_from_others(client, inject_run_failure):
    """批次中一条运行时失败不影响其他条目：partial_failed，其余成功且结果可取。"""
    final, res = _run_batch_and_get_result(client, SHORT_BASE, [1, 2, 3, 4])
    print(f"[isolate] 一批 4 条其中 seed=3 运行失败 -> {final['status']} counts={final['counts']}")
    assert final["status"] == "partial_failed"
    assert final["counts"]["failed"] == 1
    assert final["counts"]["succeeded"] == 3
    by_seed = {i["params"]["seed"]: i for i in res["items"]}
    bad = by_seed[3]
    assert bad["status"] == "failed"
    assert bad["error_reason"] == "internal_error"
    assert bad["final_state"] is None
    for s in (1, 2, 4):
        assert by_seed[s]["status"] == "succeeded"
        assert by_seed[s]["final_state"] is not None
        print(f"[isolate] seed={s} 未受失败条目影响，checksum={by_seed[s]['checksum']:.6f}")


def test_resubmit_same_batch_is_pointwise_reproducible(client, inject_run_failure):
    """同批同样参数+种子重提：成功条目逐点一致，失败条目同样失败、同原因。"""
    seeds = [1, 2, 3, 4]
    f1, r1 = _run_batch_and_get_result(client, SHORT_BASE, seeds)
    f2, r2 = _run_batch_and_get_result(client, SHORT_BASE, seeds)
    assert f1["counts"] == f2["counts"]
    for a, b in zip(r1["items"], r2["items"]):
        assert a["params"] == b["params"]
        assert a["status"] == b["status"]
        assert a["error_reason"] == b["error_reason"]
        if a["status"] == "succeeded":
            assert a["final_state"] == b["final_state"]
            assert a["checksum"] == b["checksum"]
    print("[reproduce] 整批重提：4 条（含 1 条 internal_error）状态、原因、"
          "成功条目 final_state/checksum 全部逐点一致")


def test_cancelled_earlier_batch_does_not_affect_reproducibility(client, inject_run_failure):
    """上一批里有条目被取消，不影响其余组合在新批次中的逐点复现。"""
    seeds = [1, 2, 3, 4]
    # 基线批次：全部跑完（seed=3 失败）
    _, base_res = _run_batch_and_get_result(client, SHORT_BASE, seeds)
    baseline_by_seed = {i["params"]["seed"]: i for i in base_res["items"]}

    # 第二批：用单任务占满 worker，让批次条目都排队，取消其中一条后再放开
    blockers = [client.post("/simulations", json={**LONG_BASE, "seed": 200 + i}).json()["task_id"]
                for i in range(2)]
    bid = submit_batch(client, SHORT_BASE, [{"field": "seed", "values": seeds}]).json()["batch_id"]
    time.sleep(0.3)
    v = client.get(f"/batches/{bid}").json()
    victim = next(i["item_id"] for i in v["items"] if i["status"] == "queued")
    client.post(f"/batches/{bid}/items/{victim}/cancel")
    print(f"[reproduce] 第二批在排队阶段取消条目 {victim}，随后放开容量")
    for t in blockers:
        client.post(f"/simulations/{t}/cancel")

    final = wait_batch(client, bid)
    res = client.get(f"/batches/{bid}/result").json()
    assert final["counts"]["cancelled"] == 1
    for item in res["items"]:
        if item["status"] == "succeeded":
            ref = baseline_by_seed[item["params"]["seed"]]
            assert ref["status"] == "succeeded"
            assert item["final_state"] == ref["final_state"]
            assert item["checksum"] == ref["checksum"]
    print("[reproduce] 上一批存在被取消条目，其余成功组合与基线批次逐点一致；"
          f"counts={final['counts']}")


def test_batch_item_matches_single_task_endpoints(client):
    """批次条目派生出的 task_id 在单任务入口查到的状态/结果与批次入口一致。"""
    bid = submit_batch(client, SHORT_BASE,
                       [{"field": "seed", "values": [11, 12]}]).json()["batch_id"]
    final = wait_batch(client, bid)
    assert final["status"] == "succeeded"
    res = client.get(f"/batches/{bid}/result").json()
    for item in res["items"]:
        tid = item["task_id"]
        single_view = client.get(f"/simulations/{tid}").json()
        single_res = client.get(f"/simulations/{tid}/result").json()
        assert single_view["status"] == item["status"] == "succeeded"
        assert single_res["checksum"] == item["checksum"]
        assert single_res["final_state"] == item["final_state"]
    print("[consistency] 批次内 2 条 task_id 在单任务入口与批次入口状态/结果完全一致")
