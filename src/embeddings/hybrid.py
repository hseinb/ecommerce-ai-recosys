from typing import Dict, List, Optional
import numpy as np
import polars as pl
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA


class HybridEmbeddingBuilder:
    """
    Constructs hybrid item embeddings by fusing:
    1. Graph/Basket Latent Vectors (Item2Vec)
    2. Categorical Hierarchy Features (One-Hot / Dense)
    3. Price Tier & Margin Numerical Features
    4. Temporal / Demand Velocity Metrics
    """

    def __init__(self, target_dim: int = 64):
        self.target_dim = target_dim
        self.cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.num_scaler = StandardScaler()
        self.pca = PCA(n_components=target_dim, random_state=42)

    def build_hybrid_vectors(
        self,
        items_df: pl.DataFrame,
        item2vec_dict: Dict[str, np.ndarray],
        gemini_embeddings_dict: Optional[Dict[str, np.ndarray]] = None,
        item_col: str = "item_id",
        category_col: str = "category_name",
        price_col: str = "retail_price",
        margin_col: str = "margin",
    ) -> Dict[str, np.ndarray]:
        """
        Fuses item attributes with Item2Vec vectors and Gemini semantic embeddings, projecting to target_dim.
        """
        pdf = items_df.to_pandas()
        item_ids = pdf[item_col].astype(str).tolist()

        # 1. Categorical features
        categories = pdf[[category_col]].fillna("Unknown")
        cat_features = self.cat_encoder.fit_transform(categories)

        # 2. Numerical features (Price & Margin)
        num_features = pdf[[price_col, margin_col]].fillna(0.0).values
        scaled_num_features = self.num_scaler.fit_transform(num_features)

        # 3. Item2Vec vectors
        item2vec_dim = next(iter(item2vec_dict.values())).shape[0] if item2vec_dict else 32
        item2vec_features = np.zeros((len(item_ids), item2vec_dim), dtype=np.float32)

        for i, i_id in enumerate(item_ids):
            if i_id in item2vec_dict:
                item2vec_features[i] = item2vec_dict[i_id]
            else:
                item2vec_features[i] = np.random.normal(0, 0.01, item2vec_dim)

        # 4. Gemini Semantic Embeddings (gemini-embedding-2)
        feature_blocks = [item2vec_features, cat_features, scaled_num_features]
        if gemini_embeddings_dict:
            gemini_dim = next(iter(gemini_embeddings_dict.values())).shape[0]
            gemini_features = np.zeros((len(item_ids), gemini_dim), dtype=np.float32)
            for i, i_id in enumerate(item_ids):
                if i_id in gemini_embeddings_dict:
                    gemini_features[i] = gemini_embeddings_dict[i_id]
            feature_blocks.insert(0, gemini_features)

        # Concatenate all feature representations
        fused = np.hstack(feature_blocks)

        # Dimensionality Reduction / Projection to target_dim
        if fused.shape[1] > self.target_dim:
            embeddings_matrix = self.pca.fit_transform(fused)
        else:
            # Pad if smaller than target_dim
            padding = np.zeros((fused.shape[0], self.target_dim - fused.shape[1]))
            embeddings_matrix = np.hstack([fused, padding])

        # L2-normalize vectors for Cosine Similarity
        norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normalized_matrix = embeddings_matrix / norms

        return {i_id: normalized_matrix[i] for i, i_id in enumerate(item_ids)}
