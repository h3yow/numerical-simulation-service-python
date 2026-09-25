"""本地数值仿真任务服务：一维对流-扩散方程的异步任务式仿真 API。"""
from __future__ import annotations

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.models import (
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

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=0)
