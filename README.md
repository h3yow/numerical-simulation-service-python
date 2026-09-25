# 本地数值仿真任务服务

基于 FastAPI 的本地数值仿真服务：以任务形式异步执行**一维对流-扩散方程**的时间步进仿真，
支持提交、进度查询、取消与结果获取，并对并发数与资源占用设有明确上限。

## 计算模型与稳定性约束

求解 PDE（显式 FTCS 格式）：

```
du/dt + c * du/dx = D * d2u/dx2,   x in [0, L]
u_new[i] = u[i] + r*(u[i+1] - 2*u[i] + u[i-1]) - (cfl/2)*(u[i+1] - u[i-1])
r   = D*dt/dx^2   （扩散数）
cfl = c*dt/dx     （对流 CFL 数）
```

请求参数（`POST /simulations`）：

| 字段 | 含义 | 约束 |
|---|---|---|
| `nx` | 空间网格点数 | 3 ~ 10000 |
| `dt` | 时间步长 | 正的有限值 |
| `num_steps` | 时间步数 | 1 ~ 200000，且 `nx*num_steps <= 2e7` |
| `diffusion` | 扩散系数 D | >= 0 |
| `advection` | 对流速度 c | 有限值，默认 0 |
| `domain_length` | 区间长度 L | > 0，默认 1 |
| `boundary` | 边界条件 | `periodic` / `dirichlet`（u=0） |
| `initial` | 初始条件 | `gaussian` / `step` / `random` |
| `noise_amplitude` | 叠加在初值上的随机扰动幅度 | >= 0 |
| `seed` | 随机种子 | 含随机成分（`initial=random` 或 `noise_amplitude>0`）时必填 |

**稳定性约束**（不满足即拒绝，HTTP 422 + `unstable`）：

- `r = D*dt/dx^2 <= 0.5`
- `|cfl| = |c|*dt/dx <= 1`
- `D > 0` 时 `cfl^2 <= 2*r`

**可复现性**：相同 `seed` 与参数必然得到逐点一致的结果（结果中含 `checksum` 便于断言）。

## 任务状态、并发与取消语义

任务状态机：`queued -> running -> succeeded / failed / cancelled`。

- **并发上限**：最多 `max_concurrent`（默认 2）个任务同时执行，其余排队；
  队列上限 `max_queue`（默认 16），队列满时提交返回 HTTP 429 + `resource_limit`。
- **取消**（`POST /simulations/{id}/cancel`）幂等：
  - 排队中的任务：直接置为 `cancelled`，不会被执行；
  - 运行中的任务：求解器在固定步数的安全点退出，置为 `cancelled`，不产生结果；
  - 终态任务：无任何影响，重复取消不报错、不改变状态。
- **一致性**：状态查询（`GET /simulations/{id}`）与结果查询
  （`GET /simulations/{id}/result`）读取同一份内部状态；任务未到终态时取结果返回 409。

### 错误原因（`error_reason`，可区分）

| 值 | 含义 | 提交时 HTTP 码 |
|---|---|---|
| `invalid_params` | 参数非法（范围/组合，如缺 seed） | 400 |
| `unstable` | 违反数值稳定性约束 | 422 |
| `resource_limit` | 超出资源上限（网格/步数/队列满） | 429 |
| `cancelled` | 任务被取消 | -（任务终态） |
| `internal_error` | 求解内部错误（含运行中发散） | -（任务终态） |

非法/不稳定/超资源的参数在**提交时同步拒绝**，不会创建任务、不会返回发散或未完成的结果。

## API 一览

- `POST /simulations` — 提交仿真，201 返回任务视图（含 `task_id`）
- `GET /simulations/{task_id}` — 查询状态与进度（`progress` 0~1）
- `POST /simulations/{task_id}/cancel` — 取消任务（幂等）
- `GET /simulations/{task_id}/result` — 获取结果（末态向量 + checksum）；未终态返回 409

## 本地验证

```bash
uv sync                 # 安装依赖
uv run pytest tests/ -q -s   # 运行测试（-s 可看到关键参数与判定依据日志）
uv run python main.py        # 启动服务（127.0.0.1，随机端口）
```

测试覆盖三类场景：

- `tests/test_concurrency.py` — 并发提交与排队上限、队列满拒绝、取消排队/运行中任务、
  重复取消幂等、终态任务取消无效果、多线程并发提交+取消后的状态一致性；
- `tests/test_validation.py` — 非法参数（400/`invalid_params`）、不稳定参数
  （422/`unstable`）、超资源上限（429/`resource_limit`）三类原因可区分；
- `tests/test_reproducibility.py` — 同种子结果逐点一致、不同种子结果不同、
  状态与结果两个查询入口一致。
