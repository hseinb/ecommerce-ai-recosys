from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import polars as pl
from loguru import logger


class UserPersonalizedRecommender:
    """
    Builds personalized recommendations by computing time-decay weighted
    user preference vectors from purchase history.
    """

    def __init__(self, embeddings: Dict[str, np.ndarray], half_life_days: float = 30.0):
        self.embeddings = embeddings
        self.half_life_days = half_life_days
        self.decay_constant = np.log(2) / self.half_life_days
        self.item_ids = list(embeddings.keys())
        self.item_matrix = np.array([embeddings[i] for i in self.item_ids], dtype=np.float32)

    def build_user_vector(
        self,
        user_transactions: pl.DataFrame,
        item_col: str = "item_id",
        timestamp_col: str = "timestamp",
        reference_time: Optional[datetime] = None,
    ) -> Optional[np.ndarray]:
        """
        Aggregates items purchased by the user, weighted by time decay.
        """
        if user_transactions.is_empty():
            return None

        ref_time = reference_time or datetime.now()
        item_dim = next(iter(self.embeddings.values())).shape[0] if self.embeddings else 64
        weighted_vector = np.zeros(item_dim, dtype=np.float32)
        total_weight = 0.0

        for row in user_transactions.iter_rows(named=True):
            item_id = str(row[item_col])
            tx_time = row[timestamp_col]
            
            if item_id not in self.embeddings:
                continue

            days_diff = max(0.0, (ref_time - tx_time).total_seconds() / 86400.0) if isinstance(tx_time, datetime) else 0.0
            weight = np.exp(-self.decay_constant * days_diff)

            weighted_vector += weight * self.embeddings[item_id]
            total_weight += weight

        if total_weight > 0:
            user_vec = weighted_vector / total_weight
            norm = np.linalg.norm(user_vec)
            return user_vec / norm if norm > 0 else user_vec
        return None

    def recommend_for_user(
        self,
        customer_id: str,
        user_history: pl.DataFrame,
        top_k: int = 10,
        exclude_already_purchased: bool = True,
        stock_map: Optional[Dict[str, int]] = None,
    ) -> List[Dict[str, float]]:
        """
        Generates top-K recommendations for a specific customer.
        """
        user_vec = self.build_user_vector(user_history)
        if user_vec is None:
            return []

        # Cosine similarity against catalog
        scores = np.dot(self.item_matrix, user_vec)
        ranked_indices = np.argsort(-scores)

        purchased_items = set(user_history["item_id"].cast(pl.Utf8).to_list()) if exclude_already_purchased else set()

        results = []
        for idx in ranked_indices:
            candidate_id = self.item_ids[idx]
            if candidate_id in purchased_items:
                continue
            if stock_map and stock_map.get(candidate_id, 0) <= 0:
                continue

            results.append({
                "customer_id": customer_id,
                "item_id": candidate_id,
                "predicted_score": float(scores[idx]),
                "rank": len(results) + 1,
            })

            if len(results) >= top_k:
                break

        return results
