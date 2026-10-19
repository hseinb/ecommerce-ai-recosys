from typing import List, Dict, Optional
import numpy as np
from loguru import logger

try:
    from gensim.models import Word2Vec
except ImportError:
    Word2Vec = None


class Item2VecModel:
    """
    Learns item co-purchase latent embeddings using Skip-Gram with Negative Sampling.
    Each transaction basket is treated as a sentence, and item IDs as tokens.
    """

    def __init__(
        self,
        vector_size: int = 64,
        window: int = 5,
        min_count: int = 1,
        workers: int = 4,
        epochs: int = 30,
        sg: int = 1,  # 1 for Skip-Gram, 0 for CBOW
        negative: int = 10,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.sg = sg
        self.negative = negative
        self.model: Optional[Word2Vec] = None

    def fit(self, baskets: List[List[str]]) -> "Item2VecModel":
        if Word2Vec is None:
            raise ImportError("gensim is not installed. Please install gensim to train Item2Vec.")

        logger.info(f"Training Item2Vec model on {len(baskets)} transaction baskets (vector_size={self.vector_size})...")
        
        self.model = Word2Vec(
            sentences=baskets,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            negative=self.negative,
            epochs=self.epochs,
            seed=42,
        )
        logger.info(f"Item2Vec training complete. Vocabulary size: {len(self.model.wv.key_to_index)} items.")
        return self

    def get_embedding(self, item_id: str) -> Optional[np.ndarray]:
        if self.model and item_id in self.model.wv:
            return self.model.wv[item_id]
        return None

    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        if not self.model:
            return {}
        return {item_id: self.model.wv[item_id] for item_id in self.model.wv.key_to_index}
