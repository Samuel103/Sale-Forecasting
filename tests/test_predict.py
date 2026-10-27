import pickle
from unittest.mock import Mock

import numpy as np
import polars as pl
import pytest

from sale_forecasting.data.preprocessing import rename_columns
from sale_forecasting.model.predict import load_model, predict
from sale_forecasting.model.training import fit_category_mappings


@pytest.fixture
def raw_data():
    return pl.DataFrame({
        "id": [30, 10, 20],
        "store_nbr": [2, 1, 2],
        "family": ["BEVERAGES", "GROCERY", "BEVERAGES"],
        "onpromotion": [1, 0, 0],
        "date": ["2026-01-03", "2026-01-01", "2026-01-10"],
    })


def test_predict_applies_horizon_and_postprocessing(raw_data, monkeypatch):
    monkeypatch.setattr("sale_forecasting.model.predict.inference_days", 7)
    model = Mock()
    model.category_mappings_ = fit_category_mappings(rename_columns(raw_data))
    model.predict.return_value = np.array([12.5, 8.0])
    result = predict(raw_data, model=model)

    assert result.to_dict(as_series=False) == {
        "id": [30, 10], "customer_id": [2, 1],
        "product": ["BEVERAGES", "GROCERY"],
        "date": ["2026-01-03", "2026-01-01"], "sales": [12.5, 8.0],
    }
    assert model.predict.call_args.args[0].shape == (2, 7)


def test_empty_prediction_has_readable_schema(raw_data):
    model = Mock()
    model.category_mappings_ = fit_category_mappings(rename_columns(raw_data))
    result = predict(raw_data.head(0), model=model)
    assert result.columns == ["id", "customer_id", "product", "date", "sales"]
    assert result.is_empty()
    assert result.schema["sales"] == pl.Float64
    model.predict.assert_not_called()


def test_load_model_requires_saved_category_mappings(tmp_path):
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps({}))
    with pytest.raises(ValueError, match="category mappings"):
        load_model(path)
