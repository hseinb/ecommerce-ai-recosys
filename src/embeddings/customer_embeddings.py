from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import polars as pl
from loguru import logger


class CustomerEmbeddingBuilder:
    """
    Constructs normalized 64-dimensional customer embeddings in the same latent metric space
    as item embeddings by aggregating historical purchases with exponential time decay and volume weighting.
    """

    def __init__(self, half_life_days: float = 30.0):
        self.half_life_days = half_life_days
        self.decay_constant = np.log(2) / max(1.0, self.half_life_days)

    def build_all_customer_embeddings(
        self,
        transactions_df: pl.DataFrame,
        item_embeddings: Dict[str, np.ndarray],
        customer_col: str = "customer_id",
        item_col: str = "item_id",
        timestamp_col: str = "timestamp",
        quantity_col: str = "quantity",
        reference_time: Optional[datetime] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Builds L2-normalized customer profile vectors for every customer with transactions.
        
        Formula for customer c:
          u_c = sum( w_i * v_i ) / || sum( w_i * v_i ) ||_2
          where w_i = exp( -lambda * delta_t ) * quantity_i
        """
        ref_time = reference_time or datetime.now()
        item_dim = next(iter(item_embeddings.values())).shape[0] if item_embeddings else 64

        customer_embeddings: Dict[str, np.ndarray] = {}

        # Group transactions by customer
        grouped = transactions_df.group_by(customer_col)

        for (cust_id,), group in grouped:
            cust_str = str(cust_id)
            weighted_vector = np.zeros(item_dim, dtype=np.float32)
            total_weight = 0.0

            for row in group.iter_rows(named=True):
                i_id = str(row[item_col])
                if i_id not in item_embeddings:
                    continue

                tx_time = row[timestamp_col]
                qty = float(row.get(quantity_col, 1.0))

                days_diff = max(0.0, (ref_time - tx_time).total_seconds() / 86400.0) if isinstance(tx_time, datetime) else 0.0
                recency_weight = np.exp(-self.decay_constant * days_diff)
                effective_weight = recency_weight * max(1.0, qty)

                weighted_vector += effective_weight * item_embeddings[i_id]
                total_weight += effective_weight

            if total_weight > 0:
                norm = np.linalg.norm(weighted_vector)
                if norm > 0:
                    customer_embeddings[cust_str] = (weighted_vector / norm).astype(np.float32)
                else:
                    customer_embeddings[cust_str] = weighted_vector
            else:
                # Cold start customer fallback
                customer_embeddings[cust_str] = np.zeros(item_dim, dtype=np.float32)

        logger.info(f"Generated embeddings for {len(customer_embeddings)} customers.")
        return customer_embeddings
