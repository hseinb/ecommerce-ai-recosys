import numpy as np
import polars as pl


def add_temporal_cyclical_features(df: pl.DataFrame, timestamp_col: str = "timestamp") -> pl.DataFrame:
    """
    Extracts week_of_year, month, season, and quarter from timestamp_col and computes
    cyclical sine and cosine transformations.
    """
    return (
        df.with_columns([
            pl.col(timestamp_col).dt.week().alias("week_of_year"),
            pl.col(timestamp_col).dt.month().alias("month"),
            pl.col(timestamp_col).dt.quarter().alias("quarter"),
            pl.col(timestamp_col).dt.year().alias("year"),
        ])
        .with_columns([
            pl.when(pl.col("month").is_in([12, 1, 2]))
            .then(pl.lit("Winter"))
            .when(pl.col("month").is_in([3, 4, 5]))
            .then(pl.lit("Spring"))
            .when(pl.col("month").is_in([6, 7, 8]))
            .then(pl.lit("Summer"))
            .otherwise(pl.lit("Fall"))
            .alias("season"),
            (np.sin(2 * np.pi * pl.col("week_of_year") / 52.0)).alias("sin_week"),
            (np.cos(2 * np.pi * pl.col("week_of_year") / 52.0)).alias("cos_week"),
            (np.sin(2 * np.pi * pl.col("quarter") / 4.0)).alias("sin_quarter"),
            (np.cos(2 * np.pi * pl.col("quarter") / 4.0)).alias("cos_quarter"),
        ])
    )


def compute_seasonal_velocity_indices(
    df: pl.DataFrame,
    item_col: str = "item_id",
    quantity_col: str = "quantity",
    week_col: str = "week_of_year",
    quarter_col: str = "quarter",
) -> pl.DataFrame:
    """
    Calculates the seasonal sales velocity index for each item by week and quarter:
    Seasonal Index = (Sales_item_week) / (Average_Weekly_Sales_item)
    Values > 1.0 indicate high seasonal demand lift.
    """
    item_avg_weekly = (
        df.group_by([item_col, week_col, quarter_col])
        .agg(pl.col(quantity_col).sum().alias("weekly_sales"))
    )

    item_overall_avg = (
        item_avg_weekly.group_by(item_col)
        .agg(pl.col("weekly_sales").mean().alias("avg_weekly_sales"))
    )

    seasonal_indices = (
        item_avg_weekly.join(item_overall_avg, on=item_col, how="left")
        .with_columns(
            pl.when(pl.col("avg_weekly_sales") > 0)
            .then(pl.col("weekly_sales") / pl.col("avg_weekly_sales"))
            .otherwise(1.0)
            .alias("seasonal_velocity_index")
        )
    )

    return seasonal_indices


def compute_monthly_demand_forecast(
    df: pl.DataFrame,
    item_col: str = "item_id",
    quantity_col: str = "quantity",
    month_col: str = "month",
) -> pl.DataFrame:
    """
    Computes monthly demand indices across all years:
    Monthly Index = (Item Sales in Month M) / (Overall Monthly Average of Item)
    """
    monthly_sales = (
        df.group_by([item_col, month_col])
        .agg([
            pl.col(quantity_col).sum().alias("month_total_qty"),
            pl.col(quantity_col).count().alias("transaction_count"),
        ])
    )

    item_totals = (
        monthly_sales.group_by(item_col)
        .agg([
            pl.col("month_total_qty").sum().alias("item_lifetime_qty"),
            pl.col("month_total_qty").mean().alias("item_avg_monthly_qty"),
        ])
    )

    merged = (
        monthly_sales.join(item_totals, on=item_col, how="left")
        .with_columns(
            pl.when(pl.col("item_avg_monthly_qty") > 0)
            .then(pl.col("month_total_qty") / pl.col("item_avg_monthly_qty"))
            .otherwise(1.0)
            .alias("monthly_velocity_index")
        )
        .with_columns(
            pl.when(pl.col("monthly_velocity_index") >= 1.35)
            .then(pl.lit("PEAK_DEMAND"))
            .when(pl.col("monthly_velocity_index") >= 1.10)
            .then(pl.lit("HIGH_DEMAND"))
            .when(pl.col("monthly_velocity_index") >= 0.85)
            .then(pl.lit("NORMAL_DEMAND"))
            .otherwise(pl.lit("LOW_SEASON"))
            .alias("demand_tier")
        )
    )

    return merged


def compute_seasonal_period_forecast(
    df: pl.DataFrame,
    item_col: str = "item_id",
    quantity_col: str = "quantity",
    season_col: str = "season",
) -> pl.DataFrame:
    """
    Computes seasonal demand indices (Winter, Spring, Summer, Fall) across all years:
    Seasonal Index = (Sales in Season / Total Item Sales) * 4.0
    """
    seasonal_sales = (
        df.group_by([item_col, season_col])
        .agg([
            pl.col(quantity_col).sum().alias("season_total_qty"),
            pl.col(quantity_col).count().alias("transaction_count"),
        ])
    )

    item_totals = (
        seasonal_sales.group_by(item_col)
        .agg([
            pl.col("season_total_qty").sum().alias("item_lifetime_qty"),
            (pl.col("season_total_qty").sum() / 4.0).alias("expected_season_qty"),
        ])
    )

    merged = (
        seasonal_sales.join(item_totals, on=item_col, how="left")
        .with_columns(
            pl.when(pl.col("expected_season_qty") > 0)
            .then(pl.col("season_total_qty") / pl.col("expected_season_qty"))
            .otherwise(1.0)
            .alias("seasonal_velocity_index")
        )
        .with_columns(
            pl.when(pl.col("seasonal_velocity_index") >= 1.30)
            .then(pl.lit("PEAK_SEASON"))
            .when(pl.col("seasonal_velocity_index") >= 1.10)
            .then(pl.lit("HIGH_SEASON"))
            .when(pl.col("seasonal_velocity_index") >= 0.85)
            .then(pl.lit("BALANCED"))
            .otherwise(pl.lit("OFF_SEASON"))
            .alias("season_tier")
        )
    )

    return merged
