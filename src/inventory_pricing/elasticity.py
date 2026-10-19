from typing import Dict, List, Any, Optional
import numpy as np
import polars as pl
from sklearn.cluster import KMeans
import statsmodels.api as sm
from loguru import logger


class PriceElasticityEngine:
    """
    Computes item clusters in spatial latent embedding space,
    and estimates Price Elasticity of Demand (PED) per cluster and per item
    using log-log OLS econometric regression:
      ln(Quantity) = beta_0 + epsilon_p * ln(Price) + error
    """

    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")

    def cluster_items(self, embeddings_dict: Dict[str, np.ndarray]) -> Dict[str, int]:
        """
        Clusters item embeddings into latent product micro-segments.
        """
        item_ids = list(embeddings_dict.keys())
        matrix = np.array([embeddings_dict[i] for i in item_ids], dtype=np.float32)

        # Fit KMeans
        cluster_labels = self.kmeans.fit_predict(matrix)
        return {item_id: int(cluster_labels[idx]) for idx, item_id in enumerate(item_ids)}

    def estimate_cluster_elasticity(
        self,
        transactions_df: pl.DataFrame,
        cluster_map: Dict[str, int],
        item_col: str = "item_id",
        price_col: str = "retail_price",
        quantity_col: str = "quantity",
    ) -> Dict[int, Dict[str, Any]]:
        """
        Runs log-log regression per cluster to compute price elasticity beta (PED).
        """
        # Map cluster ID to transactions
        pdf = transactions_df.to_pandas()
        pdf["cluster_id"] = pdf[item_col].astype(str).map(cluster_map)

        # Filter valid positive prices and quantities
        valid_df = pdf[(pdf[price_col] > 0) & (pdf[quantity_col] > 0) & (pdf["cluster_id"].notnull())].copy()
        valid_df["log_price"] = np.log(valid_df[price_col])
        valid_df["log_qty"] = np.log(valid_df[quantity_col])

        cluster_results = {}

        for cluster_id in range(self.n_clusters):
            subset = valid_df[valid_df["cluster_id"] == cluster_id]
            if len(subset) < 10 or subset["log_price"].nunique() < 2:
                # Insufficient price variation; fallback to inelastic default
                cluster_results[cluster_id] = {
                    "ped": -0.8,
                    "r_squared": 0.0,
                    "p_value": 1.0,
                    "recommendation": "MAINTAIN",
                }
                continue

            X = sm.add_constant(subset["log_price"])
            y = subset["log_qty"]

            try:
                model = sm.OLS(y, X).fit()
                ped = float(model.params["log_price"])
                p_val = float(model.pvalues["log_price"])
                r2 = float(model.rsquared)

                if ped < -1.2:
                    action = "MARKDOWN_PROMOTE"  # Elastic: lowering price increases unit sales & total revenue
                elif -1.0 <= ped < -0.1:
                    action = "MARGIN_OPTIMIZE"   # Inelastic: price increases preserve sales & maximize margin
                else:
                    action = "MAINTAIN"

                cluster_results[cluster_id] = {
                    "ped": round(ped, 3),
                    "r_squared": round(r2, 3),
                    "p_value": round(p_val, 4),
                    "recommendation": action,
                }
            except Exception as e:
                logger.warning(f"Failed OLS regression for cluster {cluster_id}: {e}")
                cluster_results[cluster_id] = {
                    "ped": -1.0,
                    "r_squared": 0.0,
                    "p_value": 1.0,
                    "recommendation": "MAINTAIN",
                }

        return cluster_results
