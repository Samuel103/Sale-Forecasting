"""JSON contracts matching the raw data accepted by the prediction service."""

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SaleInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    store_nbr: int
    family: Annotated[str, Field(min_length=1)]
    onpromotion: Annotated[int, Field(ge=0)]
    date: date


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: Annotated[list[SaleInput], Field(min_length=1)]

    @model_validator(mode="after")
    def unique_ids(self) -> "PredictRequest":
        if len({row.id for row in self.rows}) != len(self.rows):
            raise ValueError("Each row must have a unique id.")
        return self


class SalePrediction(BaseModel):
    id: int
    customer_id: int
    product: str
    date: date
    sales: float


class PredictResponse(BaseModel):
    predictions: list[SalePrediction]


class TrainResponse(BaseModel):
    status: Literal["completed"] = "completed"
