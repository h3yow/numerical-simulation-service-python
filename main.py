import uvicorn
from fastapi import FastAPI

from simulation_service.models import SimulationParameters
from simulation_service.tasks import TaskManager

app = FastAPI()
task_manager = TaskManager()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/simulations", status_code=201)
def submit_simulation(params: SimulationParameters):
    return task_manager.submit(params)


@app.get("/simulations/{task_id}")
def get_simulation_status(task_id: str):
    return task_manager.get_status(task_id)


@app.post("/simulations/{task_id}/cancel")
def cancel_simulation(task_id: str):
    return task_manager.cancel(task_id)


@app.get("/simulations/{task_id}/result")
def get_simulation_result(task_id: str):
    return task_manager.get_result(task_id)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=0)
