# Numerical Simulation Service

本地数值仿真任务服务：以一维对流-扩散方程的时间步进仿真为核心，提供异步任务
提交、进度查询、取消与结果获取。基于 FastAPI，仿真在线程池中有界并发执行。

## 计算模型与稳定性约束

求解一维对流-扩散方程

    du/dt + a * du/dx = D * d2u/dx2,   x in [0, L]

采用显式 FTCS 格式（对流项一阶迎风、扩散项中心差分），`dx = L / (N - 1)`。

请求参数（`POST /simulations`）：

| 字段 | 含义 |
| --- | --- |
| `grid_size` | 网格点数 N |
| `domain_length` | 区域长度 L |
| `dt` | 时间步长 |
| `num_steps` | 步进次数 |
| `advection_velocity` | 对流速度 a |
| `diffusion_coefficient` | 扩散系数 D（>= 0） |
| `boundary_condition` | `dirichlet` / `neumann` / `periodic` |
| `initial_condition` | `gaussian` / `step` / `random` |
| `noise_amplitude` | 叠加在初值上的随机扰动幅度（>= 0） |
| `seed` | 随机种子；涉及随机（随机初值或扰动）时必填 |

提交前进行校验，不合法的组合会被拒绝（HTTP 400，任务直接进入 `failed`，
不会返回发散或未完成的结果），拒绝原因可区分：

- `invalid_parameters`：如 `dt <= 0`、`D < 0`、涉及随机但未给 `seed`；
- `numerically_unstable`：违反显式格式稳定性约束
  - 扩散数 `d = D*dt/dx^2 <= 0.5`
  - 对流 CFL `c = |a|*dt/dx <= 1`
  - 组合判据 `2d + c <= 1`
- `resource_limit_exceeded`：超出资源上限（见下）。

运行中若解出现非有限值，任务以 `numerically_unstable` 失败，不会输出结果。

### 可复现性

所有随机性（随机初值、随机扰动）都来自以 `seed` 初始化的独立随机发生器；
相同参数与相同 `seed` 必然得到逐位一致的结果，不同 `seed` 结果不同。

## 任务状态、并发与取消语义

任务状态机：`queued -> running -> succeeded | failed | cancelled`。

- **并发上限**：最多 `max_concurrent`（默认 2）个仿真同时执行，超出的提交
  进入 FIFO 队列等待，而不是失败或无限占用资源；队列本身有界
  （`max_queued`，默认 64），队列满时提交以 `resource_limit_exceeded` 拒绝。
- **取消**：`POST /simulations/{id}/cancel` 幂等。
  - 排队中的任务直接从队列移除，标记 `cancelled`，不会被执行；
  - 运行中的任务置取消标志，求解器每步检查并尽快退出；
  - 重复取消、取消已终结任务均为无操作，返回当前状态，不会重复计算或
    状态错乱；`cancel` 返回时任务已处于终结状态。
- **一致性**：状态与进度由同一把锁保护，任意查询入口
  （`GET /simulations/{id}` 多次查询、取消返回值）观察到的终结状态一致；
  失败原因（`invalid_parameters` / `numerically_unstable` /
  `resource_limit_exceeded` / `cancelled` / `internal_error`）相互可区分。
- **结果**：仅 `succeeded` 任务可通过 `GET /simulations/{id}/result` 获取
  （其余状态返回 409）；未知任务 id 返回 404。

## 资源上限

| 限制 | 取值 |
| --- | --- |
| 网格点数 | 3 .. 10 000 |
| 步进次数 | 1 .. 100 000 |
| 总工作量 `grid_size * num_steps` | <= 20 000 000 |
| 同时运行的仿真 | 2（`TaskManager(max_concurrent=...)`） |
| 排队任务数 | 64（`TaskManager(max_queued=...)`） |

## 接口一览

- `POST /simulations`：提交仿真，201 返回任务快照；参数被拒绝时 400，
  响应体含 `error_reason` / `error_detail`。
- `GET /simulations/{id}`：查询状态与进度。
- `POST /simulations/{id}/cancel`：取消（幂等）。
- `GET /simulations/{id}/result`：获取最终结果（仅成功任务）。

## 本地验证

```bash
uv sync                 # 安装依赖
uv run pytest           # 运行测试（日志打印关键输入参数与判定依据）
uv run uvicorn main:app # 启动服务
```

测试覆盖三类场景：并发提交与取消（状态/结果一致、取消幂等）、非法或不稳定
参数（失败原因可区分、不产生结果）、同种子复现（逐位一致、异种子不同）。
