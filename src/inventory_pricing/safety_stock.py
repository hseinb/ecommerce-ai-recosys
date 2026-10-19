from typing import Dict, Any, List
import numpy as np
import polars as pl


class SafetyStockCalculator:
    """
    Calculates statistical safety stock, dynamic reorder points (ROP),
    and inventory health alerts based on daily demand velocity and supplier lead times.
    """

    def __init__(self, z_score: float = 1.65):
        """
        z_score: Standard normal distribution factor for service level
                 1.65 -> 95% service level (standard retail)
                 1.96 -> 97.5% service level
                 2.33 -> 99% service level (critical items)
        """
        self.z_score = z_score

    def calculate_item_rop(
        self,
        avg_daily_demand: float,
        std_daily_demand: float,
        lead_time_days: float,
        lead_time_std_dev: float = 0.0,
        current_stock: int = 0,
    ) -> Dict[str, Any]:
        """
        Formula:
          Safety Stock = Z * sqrt( LT * sigma_d^2 + d_bar^2 * sigma_LT^2 )
          Reorder Point = (d_bar * LT) + Safety Stock
        """
        variance_term = (lead_time_days * (std_daily_demand ** 2)) + ((avg_daily_demand ** 2) * (lead_time_std_dev ** 2))
        safety_stock = float(self.z_score * np.sqrt(max(0.0, variance_term)))
        lead_time_demand = float(avg_daily_demand * lead_time_days)
        reorder_point = float(lead_time_demand + safety_stock)

        # Classify inventory health
        if current_stock <= safety_stock:
            stock_status = "CRITICAL_SAFETY_BREACH"
        elif current_stock <= reorder_point:
            stock_status = "REORDER_NOW"
        elif reorder_point > 0 and current_stock >= (3 * reorder_point):
            stock_status = "OVERSTOCKED"
        else:
            stock_status = "OK"

        return {
            "avg_daily_demand": float(avg_daily_demand),
            "std_daily_demand": float(std_daily_demand),
            "lead_time_days": float(lead_time_days),
            "safety_stock": round(safety_stock, 2),
            "reorder_point": round(reorder_point, 2),
            "current_stock": int(current_stock),
            "stock_status": stock_status,
        }

    def batch_calculate(
        self,
        inventory_metrics_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Computes ROP and safety stock for an entire Polars DataFrame.
        """
        return inventory_metrics_df.with_columns([
            (
                self.z_score * (
                    pl.col("lead_time_days") * (pl.col("std_daily_demand") ** 2)
                ).sqrt()
            ).alias("safety_stock"),
            (
                (pl.col("avg_daily_demand") * pl.col("lead_time_days")) +
                (
                    self.z_score * (
                        pl.col("lead_time_days") * (pl.col("std_daily_demand") ** 2)
                    ).sqrt()
                )
            ).alias("reorder_point")
        ]).with_columns([
            pl.when(pl.col("current_stock") <= pl.col("safety_stock"))
            .then(pl.lit("CRITICAL_SAFETY_BREACH"))
            .when(pl.col("current_stock") <= pl.col("reorder_point"))
            .then(pl.lit("REORDER_NOW"))
            .when(pl.col("current_stock") >= (3 * pl.col("reorder_point")))
            .then(pl.lit("OVERSTOCKED"))
            .otherwise(pl.lit("OK"))
            .alias("stock_status")
        ])
