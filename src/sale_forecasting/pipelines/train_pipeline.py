"""Executable entry point for model training."""

import polars as pl

from sale_forecasting.core.config import training_data_path
from sale_forecasting.model.training import train_model


def main() -> None:
    train_model(pl.read_csv(training_data_path))


if __name__ == "__main__":
    main()
