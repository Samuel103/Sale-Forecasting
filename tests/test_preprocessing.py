import numpy as np
import polars as pl

from sale_forecasting.core.config import feature_cols
from sale_forecasting.data.preprocessing import prepare_features, rename_columns
from sale_forecasting.model.training import fit_category_mappings, prepare_for_training


def test_training_and_inference_use_same_features():
    data = pl.DataFrame(
        {
            "id": [1, 2],
            "store_nbr": [1, 2],
            "family": ["GROCERY", "BEVERAGES"],
            "onpromotion": [0, 1],
            "date": ["2026-01-01", "2026-01-02"],
            "sales": [10.0, 20.0],
        }
    )
    mappings = fit_category_mappings(rename_columns(data))
    features, target = prepare_for_training(data, mappings)
    inference_features = prepare_features(data.drop("sales"), mappings)

    assert features.columns == feature_cols
    assert target["sales"].to_list() == [10.0, 20.0]
    np.testing.assert_array_equal(features.to_numpy(), inference_features.to_numpy())


def test_unknown_categories_reuse_training_codes():
    training = pl.DataFrame({
        "id": [1], "store_nbr": [1], "family": ["GROCERY"],
        "onpromotion": [0], "date": ["2026-01-01"], "sales": [10.0],
    })
    mappings = fit_category_mappings(rename_columns(training))
    unseen = training.with_columns(pl.lit("NEW").alias("family"))
    features = prepare_features(unseen, mappings)
    assert features["type"].to_list() == [-1]
    assert mappings["type"] == {"GROCERY": 0}
