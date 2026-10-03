"""本地数值仿真任务服务：一维对流-扩散方程的异步任务式仿真 API。"""
from __future__ import annotations

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.batch import BatchValidationError
from app.models import (
    BatchRequest,
    BatchResult,
    BatchView,
    ErrorReason,
    SimulationRequest,
    SimulationResult,
    TaskStatus,
    TaskView,
    TERMINAL_STATES,
)
from app.solver import InvalidParamsError, ResourceLimitError, StabilityError
from app.tasks import TaskManager

_STATUS_CODES = {
    InvalidParamsError: 400,
    StabilityError: 422,
    ResourceLimitError: 429,
}
_REASONS = {
    InvalidParamsError: ErrorReason.INVALID_PARAMS,
    StabilityError: ErrorReason.UNSTABLE,
    ResourceLimitError: ErrorReason.RESOURCE_LIMIT,
}
# 批次提交被拒时的 HTTP 码：原因与单任务三类拒绝完全一致
_BATCH_STATUS_CODES = {
    ErrorReason.INVALID_PARAMS.value: 400,
    ErrorReason.UNSTABLE.value: 422,
    ErrorReason.RESOURCE_LIMIT.value: 429,
}


def create_app(manager: TaskManager | None = None) -> FastAPI:
    app = FastAPI(title="numerical-simulation-service")
    app.state.manager = manager or TaskManager(max_concurrent=2, max_queue=16)

    @app.exception_handler(InvalidParamsError)
    @app.exception_handler(StabilityError)
    @app.exception_handler(ResourceLimitError)
    async def validation_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=_STATUS_CODES[type(exc)],
            content={"error_reason": _REASONS[type(exc)].value, "detail": str(exc)},
        )

    @app.exception_handler(BatchValidationError)
    async def batch_validation_handler(request: Request, exc: BatchValidationError):
        # 整批在提交阶段被拒：顶层给可区分原因，invalid_combinations 逐条说明
        return JSONResponse(
            status_code=_BATCH_STATUS_CODES[exc.reason],
            content={
                "error_reason": exc.reason,
                "detail": str(exc),
                "invalid_combinations": exc.invalid_combinations,
            },
        )

    @app.get("/")
    def read_root():
        return {"Hello": "World"}

    @app.post("/simulations", status_code=201, response_model=TaskView)
    def submit(req: SimulationRequest):
        return app.state.manager.submit(req)

    @app.get("/simulations/{task_id}", response_model=TaskView)
    def status(task_id: str):
        view = app.state.manager.get(task_id)
        if view is None:
            raise HTTPException(status_code=404, detail="task not found")
        return view

    @app.post("/simulations/{task_id}/cancel", response_model=TaskView)
    def cancel(task_id: str):
        view = app.state.manager.cancel(task_id)
        if view is None:
            raise HTTPException(status_code=404, detail="task not found")
        return view

    @app.get("/simulations/{task_id}/result", response_model=SimulationResult)
    def result(task_id: str):
        res = app.state.manager.result(task_id)
        if res is None:
            raise HTTPException(status_code=404, detail="task not found")
        if res.status not in TERMINAL_STATES:
            raise HTTPException(status_code=409, detail=f"task not finished: {res.status}")
        return res

    # ---- 批量扫参 ----
    @app.post("/batches", status_code=201, response_model=BatchView)
    def submit_batch(req: BatchRequest):
        # 提交前整批原子校验在管理器内完成；不合法时抛 BatchValidationError，
        # 由异常处理器返回可区分原因，且不会受理任何组合。
        return app.state.manager.submit_batch(req)

    @app.get("/batches/{batch_id}", response_model=BatchView)
    def get_batch(batch_id: str):
        view = app.state.manager.get_batch(batch_id)
        if view is None:
            raise HTTPException(status_code=404, detail="batch not found")
        return view

    @app.get("/batches/{batch_id}/result", response_model=BatchResult)
    def get_batch_result(batch_id: str):
        # 允许随时查看：未结束组合的状态/结果字段与状态入口保持一致
        res = app.state.manager.batch_result(batch_id)
        if res is None:
            raise HTTPException(status_code=404, detail="batch not found")
        return res

    @app.post("/batches/{batch_id}/cancel", response_model=BatchView)
    def cancel_batch(batch_id: str):
        view = app.state.manager.cancel_batch(batch_id)
        if view is None:
            raise HTTPException(status_code=404, detail="batch not found")
        return view

    @app.post("/batches/{batch_id}/items/{index}/cancel", response_model=BatchView)
    def cancel_batch_item(batch_id: str, index: int):
        item = app.state.manager.cancel_batch_item(batch_id, index)
        if item is None:
            raise HTTPException(status_code=404, detail="batch or item not found")
        # 返回整批视图，便于看到单条取消对总状态/计数的影响
        view = app.state.manager.get_batch(batch_id)
        if view is None:
            raise HTTPException(status_code=404, detail="batch not found")
        return view

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=0)
