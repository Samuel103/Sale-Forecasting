"""Thin HTTP adapters for the existing pipeline and prediction service."""

import logging

import polars as pl
from fastapi import APIRouter, HTTPException, Request

from sale_forecasting.api.schemas import PredictRequest, PredictResponse, TrainResponse
from sale_forecasting.model.predict import load_model, predict
from sale_forecasting.pipelines.train_pipeline import main as train_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/train", response_model=TrainResponse)
def train(request: Request) -> TrainResponse:
    """Train on configured data, wait for completion and reload the model."""
    if not request.app.state.training_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Training is already running.")
    try:
        train_pipeline()
        request.app.state.model = load_model()
    except Exception as exc:
        logger.exception("Training failed")
        raise HTTPException(
            status_code=500, detail="Training failed. See server logs."
        ) from exc
    finally:
        request.app.state.training_lock.release()
    return TrainResponse()


@router.post("/predict", response_model=PredictResponse)
def predict_sales(payload: PredictRequest, request: Request) -> PredictResponse:
    """Predict using the model already loaded in memory."""
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="No model available. Run /train first.")
    df = pl.DataFrame([row.model_dump(mode="json") for row in payload.rows])
    predictions = predict(df, model=model)
    return PredictResponse(predictions=predictions.to_dicts())
