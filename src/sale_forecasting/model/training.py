import pickle

import lightgbm as lgbm
import mlflow
import mlflow.sklearn
import polars as pl
from lightgbm import LGBMRegressor
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from sale_forecasting.core.config import (
    CONFIG_PATH,
    categorical_cols,
    general,
    model_export_path,
    model_settings,
)
from sale_forecasting.data.preprocessing import prepare_features, rename_columns


def fit_category_mappings(df: pl.DataFrame) -> dict[str, dict[str, int]]:
    """Learn category codes from a DataFrame with renamed columns."""
    return {
        col: {value: code for code, value in enumerate(
            df[col].cast(pl.String).drop_nulls().unique().sort().to_list()
        )}
        for col in categorical_cols
    }


def prepare_for_training(
    df: pl.DataFrame,
    category_mappings: dict[str, dict[str, int]] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Separate the training target and build features with training categories."""
    if "sales" not in df.columns:
        raise ValueError("Target column 'sales' not found in DataFrame columns.")
    if category_mappings is None:
        category_mappings = fit_category_mappings(rename_columns(df))
    return prepare_features(df, category_mappings), df.select("sales")


def train_model(df: pl.DataFrame) -> LGBMRegressor:
    """
    Trains the LightGBM model using the provided DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame containing features and target variable.

    Returns:
        LGBMRegressor: The trained LightGBM model.
    """
    model = LGBMRegressor(**model_settings)
    mlflow.set_tracking_uri(general["mlflow_url"])
    mlflow.set_experiment("retailprice")

    with mlflow.start_run(run_name="lightgbm"):
        category_mappings = fit_category_mappings(rename_columns(df))
        feature, target = prepare_for_training(df, category_mappings)

        X = feature.to_numpy()
        y = target.to_numpy().ravel()

        mlflow.log_params(model.get_params())
        mlflow.log_params({
            "training_rows": len(y),
            "feature_count": len(feature.columns),
        })

        model.fit(
            X, y, feature_name=feature.columns, categorical_feature=categorical_cols
        )
        model.category_mappings_ = category_mappings

        # Explicit canvases avoid GUI managers in FastAPI worker threads,
        # without changing the global backend used by notebooks.
        for importance_type in ("gain", "split"):
            figure = Figure(figsize=(10, 6))
            FigureCanvasAgg(figure)
            try:
                lgbm.plot_importance(
                    model, ax=figure.subplots(), importance_type=importance_type
                )
                mlflow.log_figure(figure, f"charts/feature_{importance_type}.png")
            finally:
                figure.clear()

        predictions = model.predict(X)
        mlflow.log_metrics({
            "train_mae": mean_absolute_error(y, predictions),
            "train_rmse": root_mean_squared_error(y, predictions),
        })

        # Preserve the explicit model export consumed by model/predict.py.
        model_export_path.parent.mkdir(parents=True, exist_ok=True)
        with model_export_path.open("wb") as model_file:
            pickle.dump(model, model_file)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="cloudpickle",
            signature=mlflow.models.infer_signature(X[:5], predictions[:5]),
        )
        mlflow.log_artifact(str(CONFIG_PATH))

    return model

