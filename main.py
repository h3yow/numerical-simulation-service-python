"""本地数值仿真任务服务：一维对流-扩散方程的异步任务式仿真 API。"""
from __future__ import annotations

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.batch import BatchManager, BatchValidationError
from app.models import (
    BatchRequest,
    BatchResultView,
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


def create_app(
    manager: TaskManager | None = None,
    batch_manager: BatchManager | None = None,
) -> FastAPI:
    app = FastAPI(title="numerical-simulation-service")
    app.state.manager = manager or TaskManager(max_concurrent=2, max_queue=16)
    app.state.batch_manager = batch_manager or BatchManager(app.state.manager)

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
        return JSONResponse(
            status_code=exc.http_status,
            content={
                "error_reason": exc.reason.value,
                "detail": exc.detail,
                "violations": exc.violations,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(request: Request, exc: RequestValidationError):
        # 请求体结构/枚举非法（如扫描轴字段名不存在）：统一归类为 invalid_params
        return JSONResponse(
            status_code=400,
            content={
                "error_reason": ErrorReason.INVALID_PARAMS.value,
                "detail": "请求参数不合法",
                "violations": exc.errors(),
            },
        )

    @app.get("/")
    def read_root():
        return {"Hello": "World"}

    # ---- 单任务（行为保持不变） ----
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
        return app.state.batch_manager.submit(req)

    @app.get("/batches/{batch_id}", response_model=BatchView)
    def batch_status(batch_id: str):
        view = app.state.batch_manager.get(batch_id)
        if view is None:
            raise HTTPException(status_code=404, detail="batch not found")
        return view

    @app.get("/batches/{batch_id}/result", response_model=BatchResultView)
    def batch_result(batch_id: str):
        view = app.state.batch_manager.result(batch_id)
        if view is None:
            raise HTTPException(status_code=404, detail="batch not found")
        return view

    @app.post("/batches/{batch_id}/cancel", response_model=BatchView)
    def batch_cancel(batch_id: str):
        view = app.state.batch_manager.cancel(batch_id)
        if view is None:
            raise HTTPException(status_code=404, detail="batch not found")
        return view

    @app.post("/batches/{batch_id}/items/{item_id}/cancel", response_model=BatchView)
    def batch_item_cancel(batch_id: str, item_id: str):
        view = app.state.batch_manager.cancel_item(batch_id, item_id)
        if view is None:
            raise HTTPException(status_code=404, detail="batch or item not found")
        return view

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=0)
