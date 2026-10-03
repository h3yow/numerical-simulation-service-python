"""场景四：批量扫参——展开对号、上限拒绝、整批/单条取消幂等、
非法/不稳定组合提交前拦截、批次间同种子复现、与单任务共用容量。

运行：uv run pytest tests/test_batch.py -s   （-s 可看各场景判定日志）
"""
import time

import pytest
from fastapi.testclient import TestClient

import app.tasks as tasks_mod
from app.tasks import TaskManager
from main import create_app
from tests.conftest import wait_terminal

# 短任务：毫秒级，用于正常跑完的批次
SHORT_BASE = {
    "nx": 64, "dt": 1e-4, "num_steps": 300, "diffusion": 0.05,
    "advection": 0.2, "noise_amplitude": 0.01,
}
# 长任务：占住 worker，便于观察 queued/running 并及时取消
LONG_STEPS = 100_000
LONG_AXES = [
    {"field": "num_steps", "values": [LONG_STEPS, 300]},
    {"field": "seed", "values": [1, 2]},
]  # 展开顺序：0=长seed1, 1=长seed2, 2=短seed1, 3=短seed2


@pytest.fixture()
def manager():
    m = TaskManager(max_concurrent=2, max_queue=32)
    yield m
    m.shutdown()


@pytest.fixture()
def client(manager):
    with TestClient(create_app(manager)) as c:
        yield c


def _submit_batch(client, base, axes):
    r = client.post("/batches", json={"base": base, "axes": axes})
    return r


def wait_batch(client, batch_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        view = client.get(f"/batches/{batch_id}").json()
        if view["status"] == "completed":
            return view
        time.sleep(0.02)
    raise TimeoutError(f"batch {batch_id} 未在 {timeout}s 内完成")


def _cancel_all_batch(client, view):
    """测试收尾：整批取消，避免长任务拖到进程退出。"""
    client.post(f"/batches/{view['batch_id']}/cancel")


# ---------- 1) 组合展开与结果对号 ----------

def test_batch_expansion_order_and_result_matching(client):
    """笛卡尔积按「最后一轴变化最快」展开；每个结果都能对回参数与轴取值。"""
    payload_base = {**SHORT_BASE, "seed": 0}
    r = _submit_batch(client, payload_base, [
        {"field": "advection", "values": [0.1, 0.2]},
        {"field": "seed", "values": [11, 22]},
    ])
    assert r.status_code == 201, r.json()
    view = r.json()
    bid = view["batch_id"]
    print(f"[batch-expand] 批次 {bid[:8]} 展开组合数={view['total']}（预期 4）")
    assert view["total"] == 4
    expected = [
        (0, 0.1, 11), (1, 0.1, 22), (2, 0.2, 11), (3, 0.2, 22),
    ]
    for index, adv, seed in expected:
        item = view["items"][index]
        assert item["index"] == index
        assert item["axis_values"] == {"advection": adv, "seed": seed}
        assert item["params"]["advection"] == adv
        assert item["params"]["seed"] == seed
        assert item["task_id"]
    print("[batch-expand] 展开顺序与轴取值对号：",
          [(it["index"], it["axis_values"]) for it in view["items"]])

    final = wait_batch(client, bid)
    assert final["succeeded"] == 4 and final["progress"] == 1.0
    res = client.get(f"/batches/{bid}/result").json()
    assert res["status"] == "completed" and len(res["items"]) == 4
    for index, adv, seed in expected:
        it = res["items"][index]
        assert it["status"] == "succeeded"
        assert it["axis_values"] == {"advection": adv, "seed": seed}
        assert it["params"]["advection"] == adv and it["params"]["seed"] == seed
        assert len(it["final_state"]) == SHORT_BASE["nx"]
        # 状态入口与结果入口同一 task_id、同一状态
        assert it["task_id"] == final["items"][index]["task_id"]
        single = client.get(f"/simulations/{it['task_id']}").json()
        assert single["status"] == "succeeded"
        one = client.get(f"/simulations/{it['task_id']}/result").json()
        assert one["checksum"] == it["checksum"]
    print("[batch-expand] 4 条结果全部成功，结果入口的 index/轴取值/task_id 与状态入口一一对应")


# ---------- 2) 展开方式不合法：提交时拒绝 ----------

def test_batch_invalid_expansion_rejected_at_submit(client):
    cases = [
        ([{"field": "seed", "values": [1, 1]}], "同一轴内取值重复"),
        ([{"field": "seed", "values": [1]},
          {"field": "seed", "values": [2]}], "扫描轴字段重复"),
        ([{"field": "boundary", "values": [1, 2]}], "扫描不可扫描字段"),
        ([{"field": "seed", "values": []}], "取值列表为空"),
        ([{"field": "nx", "values": [10, 10.5]}], "整数字段取非整数"),
    ]
    for axes, why in cases:
        r = _submit_batch(client, SHORT_BASE, axes)
        body = r.json()
        print(f"[batch-invalid] {why}: -> http={r.status_code}, "
              f"reason={body.get('error_reason')}, detail={body.get('detail')}")
        assert r.status_code == 400, (why, body)
        assert body["error_reason"] == "invalid_params"
    print("[batch-invalid] 5 类非法展开全部在提交阶段以 400/invalid_params 拒绝，未创建任何批次")


def test_batch_non_finite_value_rejected():
    """非有限值（NaN/Infinity）无法通过 JSON 传输，但校验层同样必须拒绝。"""
    from app.batch import BatchValidationError, validate_batch
    from app.models import BatchRequest, ScanAxis
    for bad in (float("nan"), float("inf"), float("-inf")):
        req = BatchRequest(base=dict(SHORT_BASE),
                           axes=[ScanAxis(field="dt", values=[1e-4, bad])])
        with pytest.raises(BatchValidationError) as exc:
            validate_batch(req)
        assert exc.value.reason == "invalid_params"
        print(f"[batch-invalid] 扫描值 {bad!r} -> invalid_params: {exc.value}")


def test_batch_at_least_one_axis_required(client):
    """axes 为空（pydantic 层）同样在提交时 422/400 拒绝。"""
    r = client.post("/batches", json={"base": SHORT_BASE, "axes": []})
    print(f"[batch-invalid] axes 为空 -> http={r.status_code}")
    assert r.status_code == 422  # 请求体本身不满足 min_length=1


# ---------- 3) 容量上限：组合数 / 总工作量 / 排队容量，提交时拒绝且不部分受理 ----------

def test_batch_combination_count_cap_rejected(client):
    """展开组合数超过 MAX_BATCH_COMBINATIONS=256：提交即 429，整批不受理。"""
    axes = [
        {"field": "seed", "values": list(range(20))},
        {"field": "advection", "values": [round(0.01 * i, 6) for i in range(13)]},
    ]  # 20*13 = 260 > 256
    r = _submit_batch(client, SHORT_BASE, axes)
    body = r.json()
    print(f"[batch-cap] 20x13=260 组合 -> http={r.status_code}, "
          f"reason={body['error_reason']}, detail={body['detail']}")
    assert r.status_code == 429 and body["error_reason"] == "resource_limit"


def test_batch_total_work_cap_rejected(client):
    """组合数合法但总工作量超 MAX_BATCH_TOTAL_WORK：同样提交即 429。"""
    base = {"nx": 10_000, "dt": 1e-12, "num_steps": 200_000, "diffusion": 0.1}
    axes = [{"field": "seed", "values": list(range(60))}]  # 60 * 2e6 = 1.2e8 > 1e8
    r = _submit_batch(client, base, axes)
    body = r.json()
    print(f"[batch-cap] 60x(10000*200000)=1.2e8 工作量 -> http={r.status_code}, "
          f"reason={body['error_reason']}")
    assert r.status_code == 429 and body["error_reason"] == "resource_limit"


def test_batch_queue_capacity_atomic_reject(client, manager):
    """排队容量不足容纳整批时：整批拒绝，不悄悄少跑；已有/后续提交不受影响。"""
    long_base = {**SHORT_BASE, "num_steps": LONG_STEPS, "seed": 7}
    blockers = [client.post("/simulations", json=long_base).json()["task_id"]
                for _ in range(2)]  # 占满 2 个 worker，占 0 队列位（直接运行）
    # 等两个都进入 running
    deadline = time.time() + 10
    while sum(client.get(f"/simulations/{t}").json()["status"] == "running"
              for t in blockers) < 2:
        assert time.time() < deadline
        time.sleep(0.01)
    # 队列上限 32，此处造一个小容量管理器单独验证
    small = TaskManager(max_concurrent=2, max_queue=4)
    try:
        with TestClient(create_app(small)) as c2:
            b = [c2.post("/simulations", json=long_base).json()["task_id"]
                 for _ in range(2)]
            deadline = time.time() + 10
            while sum(c2.get(f"/simulations/{t}").json()["status"] == "running"
                      for t in b) < 2:
                assert time.time() < deadline
                time.sleep(0.01)
            r = _submit_batch(c2, SHORT_BASE, [
                {"field": "seed", "values": [1, 2, 3, 4, 5]}])  # 需要 5 个队列位 > 4
            body = r.json()
            print(f"[batch-cap] 队列余位 4、批次需 5 -> http={r.status_code}, "
                  f"reason={body['error_reason']}（预期 429，整批零受理）")
            assert r.status_code == 429 and body["error_reason"] == "resource_limit"
            # 整批零受理：队列里仍可放进 4 个单任务
            for seed in (10, 11):
                ok = c2.post("/simulations", json={**SHORT_BASE, "seed": seed})
                assert ok.status_code == 201
            for t in b:
                c2.post(f"/simulations/{t}/cancel")
                wait_terminal(c2, t)
    finally:
        small.shutdown()
    for t in blockers:
        client.post(f"/simulations/{t}/cancel")
        wait_terminal(client, t)


# ---------- 4) 不稳定/非法组合提交前拦截，原因逐条可区分；D=0 边界结论明确 ----------

def test_batch_unstable_combo_rejected_with_itemized_reasons(client):
    """一条轴扫到不稳定取值：整批 422 拒绝，invalid_combinations 指名下标与原因。"""
    # dt=1.0：D=0 纯对流（CFL 取决于速度），D=0.1 时扩散数爆掉
    base = {"nx": 64, "dt": 1.0, "num_steps": 10, "diffusion": 0.0,
            "advection": 0.0}
    r = _submit_batch(client, base, [
        {"field": "diffusion", "values": [0.0, 0.1]}])
    body = r.json()
    print(f"[batch-unstable] 含不稳定组合 -> http={r.status_code}, "
          f"reason={body['error_reason']}, invalid={body['invalid_combinations']}")
    assert r.status_code == 422 and body["error_reason"] == "unstable"
    inv = body["invalid_combinations"]
    assert [x["index"] for x in inv] == [1]
    assert inv[0]["axis_values"] == {"diffusion": 0.1}
    assert inv[0]["error_reason"] == "unstable"


def test_batch_invalid_combo_reason_distinct(client):
    """组合层面非法（扫到 noise>0 却无 seed）：400 + invalid_params，逐条指名。"""
    base = {"nx": 64, "dt": 1e-4, "num_steps": 10, "diffusion": 0.05}
    r = _submit_batch(client, base, [
        {"field": "noise_amplitude", "values": [0.0, 0.1]}])
    body = r.json()
    print(f"[batch-unstable] 缺 seed 的组合 -> http={r.status_code}, "
          f"reason={body['error_reason']}, invalid={body['invalid_combinations']}")
    assert r.status_code == 400 and body["error_reason"] == "invalid_params"
    assert body["invalid_combinations"][0]["index"] == 1


def test_diffusion_zero_pure_advection_boundary_is_decisive(client):
    """D=0 只留对流：CFL 满足 => 提交并跑通；CFL>1 => 提交时明确判 unstable。"""
    # 能跑：dx=1/63≈0.01587，cfl=0.5*1e-4/0.01587≈0.00315 <= 1
    ok_base = {"nx": 64, "dt": 1e-4, "num_steps": 300,
               "diffusion": 0.0, "advection": 0.5, "seed": 3}
    r = _submit_batch(client, ok_base, [
        {"field": "diffusion", "values": [0.0, 0.05]}])
    assert r.status_code == 201, r.json()
    view = r.json()
    final = wait_batch(client, view["batch_id"])
    print(f"[batch-boundary] D 扫到 0（纯对流, CFL≈0.003）: "
          f"succeeded={final['succeeded']}/2（服务结论：能跑，且与有扩散组合一样正常受理）")
    assert final["succeeded"] == 2

    # 不能跑：D=0 且 |c|*dt/dx > 1 -> unstable（纯对流同样受 CFL 约束）
    bad_base = {"nx": 64, "dt": 0.01, "num_steps": 10,
                "diffusion": 0.0, "advection": 100.0}
    r = _submit_batch(client, bad_base, [
        {"field": "diffusion", "values": [0.0, 1e-12]}])
    # 两个组合 CFL 都超限（展开方式合法、取值不重复）
    body = r.json()
    print(f"[batch-boundary] D=0 纯对流但 CFL>1 -> http={r.status_code}, "
          f"reason={body['error_reason']}（服务结论：不能跑，提交时即拦截）")
    assert r.status_code == 422 and body["error_reason"] == "unstable"
    assert len(body["invalid_combinations"]) == 2


# ---------- 5) 整批取消：排队的不执行、在跑的尽快停；重复取消幂等 ----------

def test_cancel_whole_batch_idempotent(client):
    r = _submit_batch(client, {**SHORT_BASE, "num_steps": LONG_STEPS},
                      [{"field": "seed", "values": [1, 2, 3, 4]}])
    assert r.status_code == 201, r.json()
    view = r.json()
    bid = view["batch_id"]
    # 等至少有组合开始跑（此时另有组合在排队）
    deadline = time.time() + 10
    while client.get(f"/batches/{bid}").json()["running"] == 0:
        assert time.time() < deadline
        time.sleep(0.01)
    v1 = client.post(f"/batches/{bid}/cancel").json()
    print(f"[batch-cancel] 首次整批取消: queued={v1['queued']}, running={v1['running']}, "
          f"cancelled={v1['cancelled']}")
    final = wait_batch(client, bid)
    v2 = client.post(f"/batches/{bid}/cancel").json()  # 终态后重复取消
    v3 = client.post(f"/batches/{bid}/cancel").json()
    assert final["status"] == "completed"
    assert final["cancelled"] == 4 and final["succeeded"] == 0
    assert (v2["queued"], v2["running"], v2["cancelled"]) == (0, 0, 4)
    assert (v3["queued"], v3["running"], v3["cancelled"]) == (0, 0, 4)
    res = client.get(f"/batches/{bid}/result").json()
    assert all(it["status"] == "cancelled" and it["final_state"] is None
               for it in res["items"])
    print("[batch-cancel] 全部 4 条 cancelled；重复取消不引发重复计算、状态不回退，"
          "结果入口无任何末态向量")


# ---------- 6) 单条取消幂等 + 与同批其他组合隔离 ----------

def test_cancel_single_item_idempotent_and_isolated(client):
    r = _submit_batch(client, {**SHORT_BASE, "advection": 0.2, "noise_amplitude": 0.01},
                      LONG_AXES)
    assert r.status_code == 201, r.json()
    view = r.json()
    bid = view["batch_id"]
    # index 0/1 是长任务占住 worker；2/3 在排队
    deadline = time.time() + 10
    while client.get(f"/batches/{bid}").json()["running"] < 2:
        assert time.time() < deadline
        time.sleep(0.01)
    item_view = client.post(f"/batches/{bid}/items/3/cancel").json()
    assert item_view["items"][3]["status"] == "cancelled"
    again = client.post(f"/batches/{bid}/items/3/cancel").json()
    assert again["items"][3]["status"] == "cancelled"  # 幂等
    print("[batch-item-cancel] 排队单条 index=3 两次取消后仍 cancelled，未被调度执行")
    # 未知下标 / 未知批次 -> 404
    assert client.post(f"/batches/{bid}/items/99/cancel").status_code == 404
    assert client.post("/batches/nope/items/0/cancel").status_code == 404
    # 取消两个正在跑的长任务，让 index=2 的短任务获得 worker
    for i in (0, 1):
        client.post(f"/batches/{bid}/items/{i}/cancel")
    final = wait_batch(client, bid)
    print(f"[batch-item-cancel] 终态计数: succeeded={final['succeeded']}, "
          f"cancelled={final['cancelled']}（预期 1 成功 / 3 取消）")
    assert final["succeeded"] == 1 and final["cancelled"] == 3
    res = client.get(f"/batches/{bid}/result").json()
    by_index = {it["index"]: it for it in res["items"]}
    assert by_index[2]["status"] == "succeeded" and by_index[2]["final_state"]
    for i in (0, 1, 3):
        assert by_index[i]["status"] == "cancelled" and by_index[i]["final_state"] is None
    print("[batch-item-cancel] 单条取消不影响同批其他组合：index=2 正常成功")


# ---------- 7) 单条运行失败不影响同批其他组合，状态/结果两入口一致 ----------

def test_item_failure_isolated(client, monkeypatch):
    real_run = tasks_mod.run_simulation

    def fake_run(req, progress_cb=None, should_cancel=None):
        if req.seed == 42:
            raise RuntimeError("injected divergence for seed=42")
        return real_run(req, progress_cb=progress_cb, should_cancel=should_cancel)

    monkeypatch.setattr(tasks_mod, "run_simulation", fake_run)
    r = _submit_batch(client, {**SHORT_BASE, "seed": 0},
                      [{"field": "seed", "values": [1, 42, 2]}])
    assert r.status_code == 201, r.json()
    bid = r.json()["batch_id"]
    final = wait_batch(client, bid)
    assert final["succeeded"] == 2 and final["failed"] == 1
    res = client.get(f"/batches/{bid}/result").json()
    by_index = {it["index"]: it for it in res["items"]}
    assert by_index[1]["status"] == "failed"
    assert by_index[1]["error_reason"] == "internal_error"
    assert by_index[1]["final_state"] is None
    for i in (0, 2):
        assert by_index[i]["status"] == "succeeded" and by_index[i]["final_state"]
        single = client.get(f"/simulations/{by_index[i]['task_id']}/result").json()
        assert single["checksum"] == by_index[i]["checksum"]
    print(f"[batch-isolate] seed=42 注入失败 -> failed/internal_error，"
          f"其余 2 条照常 succeeded，单任务入口校验和一致")
    # 对已失败的单条再取消是 noop
    client.post(f"/batches/{bid}/items/1/cancel")
    again = client.get(f"/batches/{bid}").json()
    assert again["failed"] == 1 and again["succeeded"] == 2


# ---------- 8) 整批同参数同种子重提：逐点一致，不受上一批取消影响 ----------

def test_batch_resubmit_same_seeds_pointwise_identical(client):
    axes = [{"field": "seed", "values": [101, 202, 303]}]

    def run_full():
        r = _submit_batch(client, SHORT_BASE, axes)
        assert r.status_code == 201, r.json()
        bid = r.json()["batch_id"]
        final = wait_batch(client, bid)
        res = client.get(f"/batches/{bid}/result").json()
        return final, res

    final_a, res_a = run_full()
    assert final_a["succeeded"] == 3

    # 第二批：提交后立刻整批取消（混入取消/可能的成功），人为制造「脏」历史
    r = _submit_batch(client, SHORT_BASE, axes)
    bid_b = r.json()["batch_id"]
    client.post(f"/batches/{bid_b}/cancel")
    wait_batch(client, bid_b)

    # 第三批：与第一批完全相同的 base/axes/种子
    final_c, res_c = run_full()
    for ia, ic in zip(res_a["items"], res_c["items"]):
        assert ia["axis_values"] == ic["axis_values"]
        assert ia["status"] == ic["status"] == "succeeded"
        assert ia["checksum"] == ic["checksum"]
        assert ia["final_state"] == ic["final_state"]  # 逐点一致
    print("[batch-reproduce] 三批同种子：第一批全成功、第二批被取消、第三批再次全成功；")
    print("[batch-reproduce] 第一批与第三批 3 条组合的 checksum 与末态向量逐点相等，"
          "上一批的取消不影响可复现性")


# ---------- 9) 批次与单任务共用同一套并发/排队容量 ----------

def test_batch_shares_concurrency_and_queue(client, manager):
    # 先提交 2 个单任务长占 worker
    blockers = [
        client.post("/simulations", json={**SHORT_BASE, "num_steps": LONG_STEPS,
                                          "seed": s}).json()["task_id"]
        for s in (901, 902)
    ]
    deadline = time.time() + 10
    while sum(client.get(f"/simulations/{t}").json()["status"] == "running"
              for t in blockers) < 2:
        assert time.time() < deadline
        time.sleep(0.01)
    r = _submit_batch(client, SHORT_BASE,
                      [{"field": "seed", "values": [1, 2, 3, 4]}])
    assert r.status_code == 201, r.json()
    bid = r.json()["batch_id"]
    time.sleep(0.1)
    pending = client.get(f"/batches/{bid}").json()
    print(f"[batch-capacity] worker 被 2 个单任务占住时批次状态={pending['status']}, "
          f"queued={pending['queued']}, running={pending['running']}（批次不得抢占）")
    assert pending["status"] == "pending"
    assert pending["queued"] == 4 and pending["running"] == 0
    # 放开单任务后批次按 max_concurrent=2 逐步跑完
    for t in blockers:
        client.post(f"/simulations/{t}/cancel")
        wait_terminal(client, t)
    final = wait_batch(client, bid)
    assert final["succeeded"] == 4
    assert manager.max_running_observed <= 2
    print(f"[batch-capacity] 单任务结束后批次 4 条全部 succeeded，"
          f"全程最大并发={manager.max_running_observed}（<=2，未绕过并发上限）")
