from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from io import BytesIO
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.backends.backend_agg import FigureCanvasAgg

from sale_forecasting.model import training
from sale_forecasting.model.predict import load_model, predict


def test_training_export_can_be_reused_for_prediction(tmp_path, monkeypatch):
    # Exercise real LightGBM fitting and serialization without a running MLflow server.
    tracking = Mock()
    tracking.start_run.return_value = nullcontext()
    monkeypatch.setattr(training, "mlflow", tracking)
    # The API trains in a worker thread: plotting must never create a GUI manager.
    monkeypatch.setattr(
        plt, "subplots", Mock(side_effect=AssertionError("GUI figure requested"))
    )

    def render_figure(figure, artifact_path):
        assert isinstance(figure.canvas, FigureCanvasAgg)
        output = BytesIO()
        figure.savefig(output, format="png")
        assert output.getvalue().startswith(b"\x89PNG\r\n\x1a\n")

    tracking.log_figure.side_effect = render_figure
    path = tmp_path / "model.pkl"
    monkeypatch.setattr(training, "model_export_path", path)
    monkeypatch.setattr(training, "model_settings", {
        "n_estimators": 2, "min_child_samples": 1, "verbosity": -1,
    })
    data = pl.DataFrame({
        "id": list(range(10)), "store_nbr": [1] * 10,
        "family": ["GROCERY"] * 10, "onpromotion": [0, 1] * 5,
        "date": ["2026-01-01"] * 10, "sales": [10.0, 20.0] * 5,
    })
    with ThreadPoolExecutor(max_workers=1) as executor:
        trained = executor.submit(training.train_model, data).result(timeout=30)
    loaded = load_model(path)
    assert loaded.category_mappings_ == trained.category_mappings_
    result = predict(data.drop("sales"), model=loaded)
    features, _ = training.prepare_for_training(data, trained.category_mappings_)
    np.testing.assert_allclose(result["sales"].to_numpy(), trained.predict(features.to_numpy()))
    tracking.log_metrics.assert_called_once()
    tracking.sklearn.log_model.assert_called_once()
    assert [call.args[1] for call in tracking.log_figure.call_args_list] == [
        "charts/feature_gain.png", "charts/feature_split.png",
    ]
