from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sale_forecasting.api.main import app


@pytest.fixture
def client(monkeypatch):
    model = Mock()
    model.category_mappings_ = {"customer_id": {"1": 0}, "type": {"GROCERY": 0}}
    model.predict.return_value = np.array([12.5])
    monkeypatch.setattr("sale_forecasting.api.main.load_model", lambda: model)
    with TestClient(app) as client:
        yield client, model


@pytest.fixture
def payload():
    return {"rows": [{
        "id": 10, "store_nbr": 1, "family": "GROCERY",
        "onpromotion": 0, "date": "2026-01-01",
    }]}


def test_predict_reuses_loaded_model(client, payload, monkeypatch):
    client, model = client
    training = Mock()
    monkeypatch.setattr("sale_forecasting.api.routes.train_pipeline", training)
    for _ in range(2):
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert response.json() == {"predictions": [{
            "id": 10, "customer_id": 1, "product": "GROCERY",
            "date": "2026-01-01", "sales": 12.5,
        }]}
    assert model.predict.call_count == 2
    training.assert_not_called()


@pytest.mark.parametrize("change", [
    {"date": "invalid"}, {"onpromotion": -1}, {"family": ""},
])
def test_predict_validates_rows(client, payload, change):
    client, model = client
    payload["rows"][0].update(change)
    assert client.post("/predict", json=payload).status_code == 422
    model.predict.assert_not_called()


def test_predict_rejects_empty_and_duplicate_rows(client, payload):
    client, _ = client
    assert client.post("/predict", json={"rows": []}).status_code == 422
    payload["rows"] *= 2
    assert client.post("/predict", json=payload).status_code == 422


def test_startup_without_model_allows_first_training(monkeypatch, payload):
    monkeypatch.setattr(
        "sale_forecasting.api.main.load_model", Mock(side_effect=FileNotFoundError)
    )
    training = Mock()
    model = Mock()
    monkeypatch.setattr("sale_forecasting.api.routes.train_pipeline", training)
    monkeypatch.setattr("sale_forecasting.api.routes.load_model", lambda: model)
    with TestClient(app) as client:
        assert client.post("/predict", json=payload).status_code == 503
        response = client.post("/train")
        assert response.status_code == 200
        assert response.json() == {"status": "completed"}
        assert app.state.model is model
    training.assert_called_once_with()


def test_training_failure_keeps_previous_model_and_releases_lock(client, monkeypatch):
    client, model = client
    training = Mock(side_effect=RuntimeError("MLflow unavailable"))
    monkeypatch.setattr("sale_forecasting.api.routes.train_pipeline", training)
    response = client.post("/train")
    assert response.status_code == 500
    assert app.state.model is model
    assert not app.state.training_lock.locked()


def test_training_rejects_concurrent_run(client, monkeypatch):
    client, _ = client
    training = Mock()
    monkeypatch.setattr("sale_forecasting.api.routes.train_pipeline", training)
    app.state.training_lock.acquire()
    try:
        assert client.post("/train").status_code == 409
    finally:
        app.state.training_lock.release()
    training.assert_not_called()
