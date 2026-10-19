from typing import Dict, List, Set, Any, Optional
from collections import defaultdict
import numpy as np
import polars as pl
from loguru import logger


class CustomerLookalikeRecommender:
    """
    Implements Customer-to-Customer (User-to-User) Collaborative Filtering via spatial customer embeddings.
    
    Workflow:
      1. Computes lookalike customer peers via cosine similarity in user latent space.
      2. Mines purchases made by similar peers that the target customer has not yet bought.
      3. Scores candidate items proportionally to peer similarity and peer interaction strength.
    """

    def __init__(self, customer_embeddings: Dict[str, np.ndarray]):
        self.customer_ids = [c for c, v in customer_embeddings.items() if np.linalg.norm(v) > 0]
        self.cust_idx_map = {cid: idx for idx, cid in enumerate(self.customer_ids)}
        
        if self.customer_ids:
            self.matrix = np.array([customer_embeddings[cid] for cid in self.customer_ids], dtype=np.float32)
        else:
            self.matrix = np.empty((0, 64), dtype=np.float32)

    def find_lookalike_customers(
        self,
        customer_id: str,
        top_k: int = 5,
        min_similarity: float = 0.20,
    ) -> List[Dict[str, Any]]:
        """
        Finds the top-K most similar lookalike customers for a given customer.
        """
        if customer_id not in self.cust_idx_map or self.matrix.shape[0] == 0:
            return []

        target_idx = self.cust_idx_map[customer_id]
        target_vec = self.matrix[target_idx]

        # Cosine similarity dot product
        similarities = np.dot(self.matrix, target_vec)
        ranked_indices = np.argsort(-similarities)

        lookalikes = []
        for idx in ranked_indices:
            peer_id = self.customer_ids[idx]
            if peer_id == customer_id:
                continue

            sim_score = float(similarities[idx])
            if sim_score < min_similarity:
                break

            lookalikes.append({
                "customer_id": customer_id,
                "lookalike_customer_id": peer_id,
                "similarity_score": round(sim_score, 4),
                "rank": len(lookalikes) + 1,
            })

            if len(lookalikes) >= top_k:
                break

        return lookalikes

    def recommend_from_similar_customers(
        self,
        customer_id: str,
        transactions_df: pl.DataFrame,
        top_k_items: int = 10,
        n_lookalikes: int = 10,
        stock_map: Optional[Dict[str, int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Recommends items purchased by lookalike customers that the target customer has not yet purchased.
        
        Item Score = sum_{peer in Lookalikes} ( similarity(target, peer) * peer_purchase_weight )
        """
        lookalikes = self.find_lookalike_customers(customer_id, top_k=n_lookalikes)
        if not lookalikes:
            return []

        # 1. Identify items target customer already purchased
        target_purchases: Set[str] = set(
            transactions_df.filter(pl.col("customer_id") == customer_id)["item_id"]
            .cast(pl.Utf8)
            .to_list()
        )

        # 2. Extract purchases by lookalike peers
        peer_sim_map = {l["lookalike_customer_id"]: l["similarity_score"] for l in lookalikes}
        peer_ids = list(peer_sim_map.keys())

        peer_tx = transactions_df.filter(pl.col("customer_id").is_in(peer_ids))
        if peer_tx.is_empty():
            return []

        # 3. Score candidate items
        item_scores = defaultdict(float)
        item_sources = defaultdict(list)

        for row in peer_tx.iter_rows(named=True):
            p_id = str(row["customer_id"])
            i_id = str(row["item_id"])

            # Skip items target already bought
            if i_id in target_purchases:
                continue

            # Skip items out of stock
            if stock_map and stock_map.get(i_id, 0) <= 0:
                continue

            sim = peer_sim_map.get(p_id, 0.0)
            qty = float(row.get("quantity", 1.0))
            score_contribution = sim * (1.0 + 0.1 * np.log1p(qty))

            item_scores[i_id] += score_contribution
            if p_id not in item_sources[i_id]:
                item_sources[i_id].append(p_id)

        if not item_scores:
            return []

        # 4. Sort and return top-K
        ranked_candidates = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:top_k_items]

        results = []
        for rank, (item_id, score) in enumerate(ranked_candidates, 1):
            results.append({
                "customer_id": customer_id,
                "item_id": item_id,
                "collaborative_score": round(float(score), 4),
                "rank": rank,
                "recommended_via_lookalike_ids": item_sources[item_id][:3],
            })

        return results
