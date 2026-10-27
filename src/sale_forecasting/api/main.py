"""FastAPI application, with one model shared between requests."""

from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI

from sale_forecasting.api.routes import router
from sale_forecasting.model.predict import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.training_lock = Lock()
    try:
        app.state.model = load_model()
    except FileNotFoundError:
        # Allow the first training to be triggered before an export exists.
        app.state.model = None
    yield
    app.state.model = None


app = FastAPI(title="Sale Forecasting", lifespan=lifespan)
app.include_router(router)
