import polars as pl


def do_post_processing(
    prediction: pl.DataFrame, test_df: pl.DataFrame
) -> pl.DataFrame:
    """Associate predictions with the customer, product and date from test.csv."""
    return prediction.join(
        test_df.select(
            "id",
            pl.col("store_nbr").alias("customer_id"),
            pl.col("family").alias("product"),
            "date",
        ),
        on="id",
        how="left",
        validate="m:1",
        maintain_order="left",
    ).select("id", "customer_id", "product", "date", "sales")
