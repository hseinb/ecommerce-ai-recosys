from typing import List, Dict, Optional, Any
import polars as pl
from loguru import logger


class SeasonalBoostingEngine:
    """
    Re-ranks candidate recommendations and computes featured item lists
    by combining latent similarity scores with seasonal sales velocity indices.
    """

    def __init__(self, alpha: float = 0.3):
        """
        alpha: Weight given to the seasonal lift multiplier.
               If alpha = 0.3, an item with 2.0x seasonal index gets a 30% score boost.
        """
        self.alpha = alpha

    def apply_seasonal_boost(
        self,
        base_recommendations: List[Dict[str, float]],
        seasonal_index_map: Dict[str, float],
    ) -> List[Dict[str, float]]:
        """
        Boosts recommendations:
          Boosted Score = Base Score * (1 + alpha * max(0, seasonal_index - 1))
        """
        boosted = []
        for rec in base_recommendations:
            item_id = rec.get("target_item_id") or rec.get("item_id")
            base_score = rec.get("similarity_score") or rec.get("predicted_score", 0.0)
            
            seasonal_idx = seasonal_index_map.get(item_id, 1.0)
            lift = max(0.0, seasonal_idx - 1.0)
            boosted_score = base_score * (1.0 + self.alpha * lift)

            new_rec = dict(rec)
            new_rec["seasonal_index"] = float(seasonal_idx)
            new_rec["boosted_score"] = float(boosted_score)
            boosted.append(new_rec)

        # Re-sort by boosted_score descending and re-assign rank
        boosted.sort(key=lambda x: x["boosted_score"], reverse=True)
        for i, rec in enumerate(boosted):
            rec["boosted_rank"] = i + 1

        return boosted

    def get_featured_items_for_period(
        self,
        seasonal_df: pl.DataFrame,
        week: int,
        quarter: int,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Ranks top featured items purely by seasonal velocity index for a target week and quarter.
        """
        period_df = (
            seasonal_df.filter(
                (pl.col("week_of_year") == week) & (pl.col("quarter") == quarter)
            )
            .sort("seasonal_velocity_index", descending=True)
            .head(top_k)
        )

        results = []
        for i, row in enumerate(period_df.iter_rows(named=True)):
            results.append({
                "week_of_year": week,
                "quarter": quarter,
                "item_id": str(row["item_id"]),
                "seasonal_velocity_index": float(row["seasonal_velocity_index"]),
                "boost_rank": i + 1,
            })
        return results
