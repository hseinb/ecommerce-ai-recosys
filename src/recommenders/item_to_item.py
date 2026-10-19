from typing import Dict, List, Tuple, Optional
import numpy as np
import polars as pl
from loguru import logger


class ItemToItemRecommender:
    """
    Retrieves top-K similar items based on vector cosine similarity,
    filtered by category and active stock status.
    """

    def __init__(self, embeddings: Dict[str, np.ndarray]):
        self.item_ids = list(embeddings.keys())
        self.item_idx_map = {item_id: i for i, item_id in enumerate(self.item_ids)}
        # Stack normalized vectors into matrix (N x D)
        self.vector_matrix = np.array([embeddings[item_id] for item_id in self.item_ids], dtype=np.float32)

    def get_similar_items(
        self,
        item_id: str,
        top_k: int = 10,
        exclude_out_of_stock: bool = True,
        stock_map: Optional[Dict[str, int]] = None,
        same_category_only: bool = False,
        category_map: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, float]]:
        """
        Calculates cosine similarities between target item and all catalog items.
        """
        if item_id not in self.item_idx_map:
            logger.warning(f"Item {item_id} not found in embedding space.")
            return []

        idx = self.item_idx_map[item_id]
        target_vec = self.vector_matrix[idx]  # Shape: (D,)

        # Dot product with normalized vectors gives Cosine Similarity
        scores = np.dot(self.vector_matrix, target_vec)

        # Sort indices descending
        ranked_indices = np.argsort(-scores)

        results = []
        target_cat = category_map.get(item_id) if category_map else None

        for rank_idx in ranked_indices:
            candidate_id = self.item_ids[rank_idx]
            if candidate_id == item_id:
                continue  # Skip self

            if exclude_out_of_stock and stock_map and stock_map.get(candidate_id, 0) <= 0:
                continue  # Skip out of stock

            if same_category_only and category_map and category_map.get(candidate_id) != target_cat:
                continue

            results.append({
                "source_item_id": item_id,
                "target_item_id": candidate_id,
                "similarity_score": float(scores[rank_idx]),
                "rank": len(results) + 1,
            })

            if len(results) >= top_k:
                break

        return results

    def batch_generate_similarities(
        self,
        top_k: int = 10,
        stock_map: Optional[Dict[str, int]] = None,
    ) -> List[Dict[str, float]]:
        """Generates top-K similarities for all items in catalog."""
        all_recs = []
        for item_id in self.item_ids:
            recs = self.get_similar_items(item_id, top_k=top_k, stock_map=stock_map)
            all_recs.extend(recs)
        return all_recs
