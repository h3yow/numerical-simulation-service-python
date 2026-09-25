"""场景二：非法参数与数值不稳定参数必须被明确拒绝，且拒绝原因可区分。"""


def test_invalid_params_rejected(client):
    cases = [
        ({"nx": 2, "dt": 1e-4, "num_steps": 10, "diffusion": 0.1}, "nx 小于下限"),
        ({"nx": 64, "dt": -1.0, "num_steps": 10, "diffusion": 0.1}, "dt 为负"),
        ({"nx": 64, "dt": 1e-4, "num_steps": 0, "diffusion": 0.1}, "num_steps 为 0"),
        ({"nx": 64, "dt": 1e-4, "num_steps": 10, "diffusion": 0.1,
          "initial": "random"}, "随机初值缺 seed"),
        ({"nx": 64, "dt": 1e-4, "num_steps": 10, "diffusion": 0.1,
          "noise_amplitude": 0.1}, "随机扰动缺 seed"),
    ]
    for payload, why in cases:
        r = client.post("/simulations", json=payload)
        body = r.json()
        print(f"[invalid] {why}: payload={payload} -> http={r.status_code}, "
              f"reason={body.get('error_reason')}, detail={body.get('detail')}")
        assert r.status_code == 400
        assert body["error_reason"] == "invalid_params"


def test_unstable_params_rejected(client):
    cases = [
        # dt 过大 -> 扩散数 r = D*dt/dx^2 > 0.5
        ({"nx": 64, "dt": 1.0, "num_steps": 10, "diffusion": 0.1}, "扩散数超限"),
        # 对流速度过大 -> CFL = |c|*dt/dx > 1
        ({"nx": 64, "dt": 0.01, "num_steps": 10, "diffusion": 0.0,
          "advection": 100.0}, "CFL 超限"),
        # cfl^2 > 2r 组合约束
        ({"nx": 64, "dt": 0.005, "num_steps": 10, "diffusion": 1e-4,
          "advection": 5.0}, "组合约束 cfl^2>2r"),
    ]
    for payload, why in cases:
        r = client.post("/simulations", json=payload)
        body = r.json()
        print(f"[unstable] {why}: payload={payload} -> http={r.status_code}, "
              f"reason={body.get('error_reason')}, detail={body.get('detail')}")
        assert r.status_code == 422
        assert body["error_reason"] == "unstable"


def test_resource_limit_rejected(client):
    cases = [
        ({"nx": 20_000, "dt": 1e-6, "num_steps": 10, "diffusion": 0.1}, "nx 超上限"),
        ({"nx": 64, "dt": 1e-6, "num_steps": 500_000, "diffusion": 0.1}, "步数超上限"),
        ({"nx": 5_000, "dt": 1e-8, "num_steps": 5_000, "diffusion": 0.1}, "总工作量超上限"),
    ]
    for payload, why in cases:
        r = client.post("/simulations", json=payload)
        body = r.json()
        print(f"[resource] {why}: payload={payload} -> http={r.status_code}, "
              f"reason={body.get('error_reason')}, detail={body.get('detail')}")
        assert r.status_code == 429
        assert body["error_reason"] == "resource_limit"


def test_error_reasons_are_distinct(client):
    """三类拒绝原因互不相同，可被调用方区分。"""
    reasons = set()
    for payload in [
        {"nx": 64, "dt": -1.0, "num_steps": 10, "diffusion": 0.1},
        {"nx": 64, "dt": 1.0, "num_steps": 10, "diffusion": 0.1},
        {"nx": 20_000, "dt": 1e-6, "num_steps": 10, "diffusion": 0.1},
    ]:
        reasons.add(client.post("/simulations", json=payload).json()["error_reason"])
    print(f"[distinct] 三类拒绝得到的 error_reason 集合: {reasons}")
    assert reasons == {"invalid_params", "unstable", "resource_limit"}
