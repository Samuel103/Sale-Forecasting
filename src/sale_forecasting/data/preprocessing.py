import polars as pl

from sale_forecasting.core.config import categorical_cols, feature_cols


def rename_columns(df: pl.DataFrame) -> pl.DataFrame:
    return df.rename({
        "id": "InvoiceNo",
        "store_nbr": "customer_id",
        "family": "type",
        "onpromotion": "promotion",
    })


def prepare_date_column(df: pl.DataFrame) -> pl.DataFrame:
    if df.schema["date"] == pl.String:
        df = df.with_columns(pl.col("date").str.to_date())
    else:
        df = df.with_columns(pl.col("date").cast(pl.Date))
    return df.with_columns(
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.day().alias("day"),
        pl.col("date").dt.weekday().alias("weekday"),
    )


def prepare_categorical_features(
    df: pl.DataFrame,
    category_mappings: dict[str, dict[str, int]],
) -> pl.DataFrame:
    """Reuse training codes; unknown categories become -1 (LightGBM missing)."""
    return df.with_columns([
        pl.col(col).cast(pl.String).replace_strict(
            category_mappings[col], default=-1, return_dtype=pl.Int32
        )
        for col in categorical_cols
    ])


def prepare_features(
    df: pl.DataFrame,
    category_mappings: dict[str, dict[str, int]],
) -> pl.DataFrame:
    """Build the same ordered features for training and inference.

    Category mappings must come from the training data, never the test data.
    """
    df = rename_columns(df)
    df = prepare_date_column(df)
    df = prepare_categorical_features(df, category_mappings)
    return df.select(feature_cols)
