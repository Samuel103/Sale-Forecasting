"""Load a trained model and predict from raw sales data."""

import pickle
from pathlib import Path

import polars as pl
from lightgbm import LGBMRegressor

from sale_forecasting.core.config import inference_days, model_export_path
from sale_forecasting.data.postprocessing import do_post_processing
from sale_forecasting.data.preprocessing import prepare_features


def load_model(path: Path = model_export_path) -> LGBMRegressor:
    """Load the selected model once, for example at API startup."""
    with path.open("rb") as model_file:
        model = pickle.load(model_file)
    if not hasattr(model, "category_mappings_"):
        raise ValueError(
            "Retrain the model with train_model to save its category mappings."
        )
    return model


def filter_inference_dates(df: pl.DataFrame) -> pl.DataFrame:
    """Keep the configured calendar horizon, preserving input row order."""
    if inference_days is None or df.is_empty():
        return df
    date = pl.col("date")
    date = date.str.to_date() if df.schema["date"] == pl.String else date.cast(pl.Date)
    return df.filter(date < date.min() + pl.duration(days=inference_days))


def predict(
    df: pl.DataFrame, model: LGBMRegressor | None = None
) -> pl.DataFrame:
    """Return id, customer_id, product, date and sales in original row order.

    Pass a model loaded at startup to reuse it across requests. Without one,
    load the explicitly configured model export. No training occurs here.
    """
    if model is None:
        model = load_model()
    if not hasattr(model, "category_mappings_"):
        raise ValueError(
            "Retrain the model with train_model to save its category mappings."
        )
    df = filter_inference_dates(df)
    features = prepare_features(df, model.category_mappings_)
    predictions = model.predict(features.to_numpy()) if df.height else []
    result = df.select("id").with_columns(
        pl.Series("sales", predictions, dtype=pl.Float64)
    )
    return do_post_processing(result, df)
