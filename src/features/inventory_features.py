import polars as pl
import numpy as np


def compute_daily_demand_metrics(
    transactions_df: pl.DataFrame,
    item_col: str = "item_id",
    date_col: str = "date",
    quantity_col: str = "quantity"
) -> pl.DataFrame:
    """
    Computes average daily demand (d_bar) and standard deviation of daily demand (sigma_d)
    for each item over the historical period.
    """
    # Daily sales aggregation per item
    daily_sales = (
        transactions_df.group_by([item_col, date_col])
        .agg(pl.col(quantity_col).sum().alias("daily_qty"))
    )

    metrics = (
        daily_sales.group_by(item_col)
        .agg([
            pl.col("daily_qty").mean().alias("avg_daily_demand"),
            pl.col("daily_qty").std().fill_null(0.0).alias("std_daily_demand"),
            pl.col("daily_qty").count().alias("active_days_count"),
            pl.col("daily_qty").sum().alias("total_units_sold"),
        ])
    )

    return metrics


def compute_turnover_velocity(
    inventory_df: pl.DataFrame,
    sales_metrics_df: pl.DataFrame,
    item_col: str = "item_id",
    stock_col: str = "current_stock"
) -> pl.DataFrame:
    """
    Calculates inventory turnover velocity and days of inventory on hand (DOH).
    DOH = current_stock / avg_daily_demand
    """
    merged = inventory_df.join(sales_metrics_df, on=item_col, how="left")
    
    return merged.with_columns([
        pl.when(pl.col("avg_daily_demand") > 0)
        .then(pl.col(stock_col) / pl.col("avg_daily_demand"))
        .otherwise(999.0)
        .alias("days_of_inventory_on_hand")
    ])
