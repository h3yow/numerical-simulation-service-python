import atexit

import uvicorn
from fastapi import FastAPI, HTTPException, Response

from simulation_service.models import SimulationParameters, TaskStatus
from simulation_service.tasks import TaskManager

app = FastAPI(title="Numerical Simulation Service")
task_manager = TaskManager(max_concurrent=2, max_queued=64)
atexit.register(task_manager.shutdown)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/simulations", status_code=201)
def submit_simulation(params: SimulationParameters, response: Response):
    snapshot = task_manager.submit(params)
    if snapshot.status == TaskStatus.FAILED:
        # Rejected before any computation: distinguishable reason in the body.
        response.status_code = 400
    return snapshot


@app.get("/simulations/{task_id}")
def get_simulation_status(task_id: str):
    snapshot = task_manager.get_status(task_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="task not found")
    return snapshot


@app.post("/simulations/{task_id}/cancel")
def cancel_simulation(task_id: str):
    snapshot = task_manager.cancel(task_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="task not found")
    return snapshot


@app.get("/simulations/{task_id}/result")
def get_simulation_result(task_id: str):
    snapshot = task_manager.get_status(task_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="task not found")
    result = task_manager.get_result(task_id)
    if result is None:
        raise HTTPException(
            status_code=409,
            detail=f"task is {snapshot.status}; result is only available when succeeded",
        )
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=0)
