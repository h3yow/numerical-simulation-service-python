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

## 批量扫参（参数轴笛卡尔积）

在单任务能力之上提供一层批量用法：给出一个**基础参数** `base` 和若干条**扫描轴**
`axes`，服务把每条轴的取值做**笛卡尔积**展开成一批普通仿真，一次提交、整批受理。
单任务的提交/查询/取消/结果接口与语义完全不变；批次中的每一条组合本质上就是一个
普通任务，进入**同一个队列**、受**同一套并发与排队容量**约束。

提交（`POST /batches`）：

```json
{
  "base": {
    "nx": 64, "dt": 0.0001, "num_steps": 300,
    "diffusion": 0.05, "advection": 0.2,
    "noise_amplitude": 0.01, "seed": 1
  },
  "axes": [
    {"field": "advection", "values": [0.1, 0.2, 0.3]},
    {"field": "seed", "values": [1, 2, 3, 4]}
  ]
}
```

上例展开为 3×4 = 12 条组合。201 返回批次视图 `BatchView`，其中 `batch_id` 即批次标识，
`items[]` 按展开顺序给出每条组合的 `index`、`task_id`、`params`（展开后的完整参数）与
`axis_values`（该组合在各轴上的取值）——任何一条结果都能据此对回它是哪组参数跑出来的。
展开顺序确定：按 `axes` 的顺序，**最后一条轴变化最快**，下标从 0 递增。

### 组合展开与容量上限规则（全部在提交时判定）

可扫描字段仅限标量数值字段：`nx`、`dt`、`num_steps`、`diffusion`、`advection`、
`domain_length`、`noise_amplitude`、`seed`（枚举/字符串字段不参与扫描）。

| 规则 | 拒绝原因 / HTTP 码 |
|---|---|
| 至少 1 条扫描轴，最多 8 条 | 400 `invalid_params` |
| 同一字段只能出现一条扫描轴（重复字段拒绝） | 400 `invalid_params` |
| 单条轴取值不能为空、不能重复、必须为有限值 | 400 `invalid_params` |
| `nx`/`num_steps`/`seed` 为整数字段，取值必须是整数 | 400 `invalid_params` |
| 单轴取值数 > 256，或展开组合数 > 256 | 429 `resource_limit` |
| 所有组合 `nx*num_steps` 之和 > 1e8 | 429 `resource_limit` |
| 排队容量不足以**整体**容纳本批 | 429 `resource_limit`（整批零受理） |
| 任意一条组合不合法 / 超资源 / 不稳定 | 400 / 429 / 422（按原因区分） |

- **原子受理**：任何一条组合校验不通过，整批在提交阶段即被拒绝，不会部分受理、
  不会先收下再慢慢失败；响应体 `invalid_combinations[]` 逐条给出非法组合的
  `index`、`axis_values`、`error_reason` 与 `detail`。每条组合走的是与单任务完全
  相同的 `check_stability` 校验，因此拒绝原因与单任务一致、可区分。
- **边界组合的明确结论**：扩散系数扫到 `0`（纯对流）是合法组合，只要
  `|cfl| = |c|*dt/dx <= 1` 就正常受理并跑通；`D=0` 时不再要求 `cfl^2 <= 2r`
  （该组合约束只在 `D > 0` 时生效）。纯对流但 CFL 超限同样在提交时判 `unstable`。

### 并发、排队与取消语义（与单任务的关系）

- **共用容量**：批次不绕过并发上限——worker 满时组合一律排队，和单任务混排在同一
  队列里按 FIFO 执行；排队容量不足时整批 429（见上），绝不悄悄少跑几条。
- **批次聚合状态**：`GET /batches/{id}` 返回总状态与计数
  （`queued/running/succeeded/failed/cancelled`）和平均 `progress`：
  - `pending`：仍有组合排队（可能同时有在跑）；
  - `running`：已无排队，但仍有组合在跑；
  - `completed`：全部到达终态（成功/失败/取消的任意混合）。
  计数之和恒等于 `total`；单条失败或取消不影响同批其他组合的进度与结果。
- **整批取消** `POST /batches/{id}/cancel`：对批内每条组合应用单任务取消规则——
  还在排队的直接置 `cancelled`、不再执行；正在跑的在求解器安全点尽快停下；
  已终态的不受影响。对已经完成的批次重复取消是 no-op，不会重复计算、状态不回退。
- **单条取消** `POST /batches/{id}/items/{index}/cancel`：只取消一条，幂等；
  返回取消后的整批视图。下标越界或批次不存在返回 404。
- **两个查询入口一致**：`GET /batches/{id}`（状态）与
  `GET /batches/{id}/result`（结果）读取同一份内部状态，每条 `item` 的 `status`、
  `error_reason`、`task_id` 一致；结果入口随时可查（未完成的组合状态如实反映）。
  每条组合同时也是普通任务，可用 `GET /simulations/{task_id}/result` 单独取结果。
- **可复现性**：展开顺序由请求决定、与时间无关；只要每条组合的完整参数与 `seed`
  相同，重提整批得到的每条结果逐点一致，即使上一批中存在被取消或失败的组合
  （失败/取消只影响那一条，不改动其他组合的输入）。

## API 一览

单任务：

- `POST /simulations` — 提交仿真，201 返回任务视图（含 `task_id`）
- `GET /simulations/{task_id}` — 查询状态与进度（`progress` 0~1）
- `POST /simulations/{task_id}/cancel` — 取消任务（幂等）
- `GET /simulations/{task_id}/result` — 获取结果（末态向量 + checksum）；未终态返回 409

批量扫参：

- `POST /batches` — 提交批次（`base` + `axes`），201 返回批次视图（含 `batch_id`）；
  展开非法/超限/组合校验不过时返回 400/429/422，响应体含 `invalid_combinations[]`
- `GET /batches/{batch_id}` — 查询整批总状态、计数、总进度与每条组合的状态/参数
- `GET /batches/{batch_id}/result` — 查询每条组合各自的结果（与状态入口一致）
- `POST /batches/{batch_id}/cancel` — 整批取消（幂等）
- `POST /batches/{batch_id}/items/{index}/cancel` — 只取消某一条（幂等）

## 本地验证

```bash
uv sync                 # 安装依赖
uv run pytest tests/ -q -s   # 运行测试（-s 可看到关键参数与判定依据日志）
uv run python main.py        # 启动服务（127.0.0.1，随机端口）
```

快速手工验证批量扫参（服务启动后）：

```bash
curl -s -X POST http://127.0.0.1:<port>/batches -H 'Content-Type: application/json' -d '{
  "base": {"nx": 64, "dt": 0.0001, "num_steps": 300, "diffusion": 0.05,
           "noise_amplitude": 0.01, "seed": 1},
  "axes": [{"field": "advection", "values": [0.1, 0.2, 0.3]},
           {"field": "seed", "values": [1, 2, 3, 4]}]
}'
# 用返回的 batch_id 查询：
curl -s http://127.0.0.1:<port>/batches/<batch_id> | python -m json.tool
curl -s http://127.0.0.1:<port>/batches/<batch_id>/result | python -m json.tool
```

测试覆盖场景：

- `tests/test_concurrency.py` — 并发提交与排队上限、队列满拒绝、取消排队/运行中任务、
  重复取消幂等、终态任务取消无效果、多线程并发提交+取消后的状态一致性；
- `tests/test_validation.py` — 非法参数（400/`invalid_params`）、不稳定参数
  （422/`unstable`）、超资源上限（429/`resource_limit`）三类原因可区分；
- `tests/test_reproducibility.py` — 同种子结果逐点一致、不同种子结果不同、
  状态与结果两个查询入口一致；
- `tests/test_batch.py` — 批量扫参：
  - 笛卡尔积展开顺序与结果按 `index`/轴取值/`task_id` 对号；
  - 重复取值/重复轴/不可扫描字段/空取值/非整数值/非有限值提交时拒绝；
  - 组合数上限、批次总工作量上限、排队容量不足整批零受理（429）；
  - 不稳定/非法组合提交前拦截并逐条给出可区分原因；`D=0` 纯对流边界组合能跑与否
    结论一致明确；
  - 整批取消（排队不执行、在跑尽快停）、重复取消/终态后取消幂等；
  - 单条取消幂等、与同批其他组合隔离；单条运行失败不影响其他组合、两入口一致；
  - 整批同参数同种子重提逐点一致（即使中间一批被取消）；
  - 批次与单任务共用并发上限与排队规则。
