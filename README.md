# 本地数值仿真任务服务

基于 FastAPI 的本地数值仿真服务：以任务形式异步执行**一维对流-扩散方程**的时间步进仿真，
支持单个仿真与**批量扫参**两种用法，均支持提交、进度查询、取消与结果获取，
并对并发数与资源占用设有明确上限。

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

## 批量扫参（参数扫描）

在一套基础参数（`base`，字段同 `SimulationRequest`）上指定若干条扫描轴（`axes`），
服务在**提交时**把各轴取值做笛卡尔积展开成一批组合，一次提交全部受理、批量执行。

```jsonc
POST /batches
{
  "base": {"nx": 64, "dt": 1e-4, "num_steps": 300,
           "diffusion": 0.1, "seed": 1, "noise_amplitude": 0.1},
  "axes": [
    {"field": "diffusion", "values": [0.0, 0.05, 0.1]},
    {"field": "seed",      "values": [1, 2, 3]}
  ]
}
// 3 x 3 = 9 个组合，返回 201 与 batch_id（批次标识）
```

可扫描字段：`nx`、`dt`、`num_steps`、`diffusion`、`advection`、`domain_length`、
`noise_amplitude`、`seed`、`boundary`、`initial`。不传 `axes` 时只跑 `base` 一条。

### 组合展开与容量上限（提交时强校验，任一不满足整批拒绝）

提交时按下列顺序校验，**原子受理**：要么全部组合被受理，要么一条都不受理，
不存在"先收下、跑起来再逐个失败"或"悄悄少跑几条"：

1. **轴结构合法**：字段必须可扫描；同一字段不能出现两条轴；每条轴 `values` 非空、
   且轴内取值不得重复（数值按数值相等去重，如 `1` 与 `1.0` 视为重复）。
   非法 -> HTTP 400 `invalid_params`。
2. **组合数上限**：展开后的组合总数不得超过 `max_combinations`（默认 **256**）。
   超限 -> HTTP 429 `resource_limit`，响应中给出实际组合数与上限。
3. **逐组合提交期校验**：每个组合都必须通过与单任务完全相同的
   参数合法性 / 资源上限 / 数值稳定性检查（包括把 `diffusion` 扫到 0 的纯对流组合：
   只要 `|cfl|<=1` 就能跑；纯对流但 `|cfl|>1` 同样在提交时以 `unstable` 拒绝）。
   任一组合不通过 -> 整批拒绝，HTTP 码与 `error_reason` 取第一条违规的类型
   （400/422/429 可区分），`violations` 数组逐条返回 `index / params / error_reason / detail`。
4. **总计算量上限**：`sum(nx*num_steps)` 不得超过 `max_total_work`（默认 **2e8**）。
   超限 -> HTTP 429 `resource_limit`，响应中给出实际总工作量与上限。

### 批次状态与结果对号

批次总体状态 `status` 如实聚合每条组合：

| 批次状态 | 含义 |
|---|---|
| `queued` | 还没有任何一条开始执行 |
| `running` | 已有条目在跑或已结束，但仍有条目未到终态 |
| `succeeded` | 全部条目成功 |
| `failed` | 全部条目终态且全部失败（无成功、无取消） |
| `cancelled` | 全部条目终态，仅由取消构成（无成功） |
| `partial_failed` | 全部终态：有成功也有失败（无取消） |
| `partial_cancelled` | 全部终态：存在被取消的条目 |

视图中还包含 `total`、按单条状态分桶的 `counts`（queued/running/succeeded/failed/cancelled）、
以及整体 `progress`（已到终态条目占比）。每个条目都带稳定的 `item_id`（如 `0000`、`0001`，
按笛卡尔积顺序编号）和该组合的**完整参数 `params`**，结果据此与参数一一对号。

- `GET /batches/{batch_id}` — 批次总状态、总进度、每条组合各自的状态；
- `GET /batches/{batch_id}/result` — 整批结果：每条组合的状态与仿真结果
  （`final_state` / `checksum` / `nx` / `num_steps`），未结束条目结果字段为 `null`。
  状态入口与结果入口读取同一份内部状态，信息始终一致。
- 每条组合派发到执行器后还会得到自己的 `task_id`，可直接用单任务接口
  （`GET /simulations/{task_id}` 等）查询，两入口状态/结果一致。

### 与现有并发、排队、取消语义的关系

- **同一套执行容量**：批次条目不是"VIP 通道"。批次内有一个派发器，仅在共享任务队列
  有空位时才把下一条组合送入同一个 `TaskManager`；`max_concurrent`（默认 2）满了就排队，
  `max_queue`（默认 16）满了批次就**等待空位而不是失败或绕过上限**。批次与普通单任务
  互相竞争同一份容量，谁先提交谁先占空位。
- **条目互不影响**：批次中某一条失败（`failed`）或被取消（`cancelled`），不影响同批
  其他组合的排队、执行与结果。
- **整批取消** `POST /batches/{batch_id}/cancel`（幂等）：
  - 还在等待派发 / 排队中的条目：直接置 `cancelled`，不再执行、不重复计算；
  - 正在运行的条目：求解器在固定步数的安全点尽快停下，置 `cancelled`，不产生结果；
  - 已到终态的批次再取消：无效果，状态不回退；重复取消结果一致。
- **单条取消** `POST /batches/{batch_id}/items/{item_id}/cancel`（幂等）：只取消指定组合，
  其余条目照常推进；对已终态条目再取消无效果；不存在的批次/条目返回 404。
- **可复现性**：同批次同样的参数与种子重新提交一次，每条结果逐点一致
  （对比 `final_state` 与 `checksum`）；即使上一批里有条目被取消或运行失败，
  也只影响那些条目本身，不影响其他组合的结果。运行中数值发散等异常归类为
  `internal_error`，在该条目的结果中可区分。

### 批量接口一览

- `POST /batches` — 提交批量扫参，201 返回批次视图（含 `batch_id`）；校验不过整批 4xx 拒绝
- `GET /batches/{batch_id}` — 批次总状态/总进度 + 每条组合状态
- `GET /batches/{batch_id}/result` — 整批结果（每组合参数对号 + 状态 + 结果）
- `POST /batches/{batch_id}/cancel` — 取消整批（幂等）
- `POST /batches/{batch_id}/items/{item_id}/cancel` — 只取消某一条（幂等）

## API 一览（单任务）

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

### 手动验证批量扫参（以启动后的服务为例）

```bash
# 提交：扩散系数 x 种子 两轴扫描，共 6 个组合
curl -s -X POST http://127.0.0.1:<port>/batches -H 'Content-Type: application/json' -d '{
  "base": {"nx": 64, "dt": 1e-4, "num_steps": 300, "diffusion": 0.1,
           "seed": 1, "noise_amplitude": 0.1},
  "axes": [{"field": "diffusion", "values": [0.0, 0.05, 0.1]},
           {"field": "seed", "values": [1, 2]}]}'
# 用返回的 batch_id 查整批状态/结果
curl -s http://127.0.0.1:<port>/batches/<batch_id>
curl -s http://127.0.0.1:<port>/batches/<batch_id>/result
# 整批取消 / 只取消某一条
curl -s -X POST http://127.0.0.1:<port>/batches/<batch_id>/cancel
curl -s -X POST http://127.0.0.1:<port>/batches/<batch_id>/items/0002/cancel
```

测试覆盖以下场景：

- `tests/test_concurrency.py` — 单任务：并发提交与排队上限、队列满拒绝、取消排队/运行中任务、
  重复取消幂等、终态任务取消无效果、多线程并发提交+取消后的状态一致性；
- `tests/test_validation.py` — 单任务：非法参数（400/`invalid_params`）、不稳定参数
  （422/`unstable`）、超资源上限（429/`resource_limit`）三类原因可区分；
- `tests/test_reproducibility.py` — 单任务：同种子结果逐点一致、不同种子结果不同、
  状态与结果两个查询入口一致；
- `tests/test_batch.py` — 批量扫参（21 个用例）：
  - 笛卡尔积展开与结果按 `params` 对号、空轴单组合、重复取值/重复轴/非法轴字段被拒；
  - 组合数上限、总计算量上限在提交时 429 拒绝、被拒批次不受理任何一条；
  - 含不稳定/超资源/需 seed/扩散=0 纯对流边界组合时整批在提交时被拦下且原因可区分；
  - 批次与单任务共用同一并发/排队（运行数不超上限、队列满时排队不失败）；
  - 整批取消、单条取消（排队/运行中）、重复取消与终态再取消的幂等；
  - 批次总状态如实反映排队/运行/结束数量，状态入口与结果入口一致；
  - 单条运行失败与单条取消不影响其他条目；
  - 整批重提同种子结果逐点一致（即使上一批有失败/被取消的条目）；
  - 批次条目与单任务入口的状态/结果一致。
