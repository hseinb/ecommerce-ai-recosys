from typing import List, Tuple
import polars as pl


def extract_transaction_baskets(
    df: pl.DataFrame,
    transaction_col: str = "transaction_id",
    item_col: str = "item_id"
) -> List[List[str]]:
    """
    Groups transactions into list of item strings (basket sequences)
    suitable for Item2Vec / Skip-Gram training.
    """
    baskets_df = (
        df.filter(pl.col(item_col).is_not_null())
        .with_columns(pl.col(item_col).cast(pl.Utf8))
        .group_by(transaction_col)
        .agg(pl.col(item_col).alias("items"))
        .filter(pl.col("items").list.len() > 1)  # Only keep multi-item baskets
    )
    
    return baskets_df["items"].to_list()


def compute_item_co_occurrence(
    df: pl.DataFrame,
    transaction_col: str = "transaction_id",
    item_col: str = "item_id"
) -> pl.DataFrame:
    """
    Computes pairwise co-occurrence frequencies between items in the same transaction basket.
    """
    items_by_tx = (
        df.select([transaction_col, item_col])
        .unique()
        .with_columns(pl.col(item_col).cast(pl.Utf8))
    )

    # Self join on transaction to get item pairs
    pairs = (
        items_by_tx.join(items_by_tx, on=transaction_col, suffix="_target")
        .filter(pl.col(item_col) != pl.col(f"{item_col}_target"))
        .group_by([item_col, f"{item_col}_target"])
        .len(name="co_occurrence_count")
    )

    return pairs
