from typing import List, Dict, Any, Optional
from loguru import logger
import numpy as np

try:
    from supabase import create_client, Client
except ImportError:
    Client = None
    create_client = None

from config.settings import settings


class SupabaseManager:
    """Manages connections and bulk operations to Supabase."""

    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        self.url = url or settings.SUPABASE_URL
        self.key = key or settings.SUPABASE_KEY
        self.client: Optional[Client] = None
        
        if self.url and self.key and create_client:
            try:
                self.client = create_client(self.url, self.key)
                logger.info("Successfully initialized Supabase client.")
            except Exception as e:
                logger.error(f"Failed to connect to Supabase: {e}")
        else:
            logger.warning("Supabase credentials not set or supabase package not loaded. Operating in offline/mock mode.")

    def upsert_items(self, items_data: List[Dict[str, Any]]) -> bool:
        """Upsert item catalog records."""
        if not self.client:
            logger.info(f"[Offline] Mock upsert {len(items_data)} items.")
            return True
        try:
            self.client.table("items").upsert(items_data).execute()
            logger.info(f"Upserted {len(items_data)} items to Supabase.")
            return True
        except Exception as e:
            logger.error(f"Error upserting items to Supabase: {e}")
            return False

    def upsert_item_embeddings(self, embeddings_data: List[Dict[str, Any]]) -> bool:
        """
        Upsert item embeddings.
        Each record should contain: {'item_id': str, 'embedding': list[float], 'cluster_id': int}
        """
        if not self.client:
            logger.info(f"[Offline] Mock upsert {len(embeddings_data)} item embeddings.")
            return True
        try:
            self.client.table("item_embeddings").upsert(embeddings_data).execute()
            logger.info(f"Upserted {len(embeddings_data)} item embeddings to Supabase.")
            return True
        except Exception as e:
            logger.error(f"Error upserting item embeddings: {e}")
            return False

    def upsert_customer_embeddings(self, embeddings_data: List[Dict[str, Any]]) -> bool:
        """
        Upsert customer spatial embeddings into pgvector.
        Each record should contain: {'customer_id': str, 'embedding': list[float], 'persona_segment_id': int}
        """
        if not self.client:
            logger.info(f"[Offline] Mock upsert {len(embeddings_data)} customer embeddings.")
            return True
        try:
            self.client.table("customer_embeddings").upsert(embeddings_data).execute()
            logger.info(f"Upserted {len(embeddings_data)} customer embeddings to Supabase.")
            return True
        except Exception as e:
            logger.error(f"Error upserting customer embeddings: {e}")
            return False

    def upsert_customer_lookalikes(self, lookalike_records: List[Dict[str, Any]]) -> bool:
        """Upsert customer lookalike pairs."""
        if not self.client:
            logger.info(f"[Offline] Mock upsert {len(lookalike_records)} lookalike pairs.")
            return True
        try:
            self.client.table("customer_lookalikes").upsert(lookalike_records).execute()
            return True
        except Exception as e:
            logger.error(f"Error upserting customer lookalikes: {e}")
            return False

    def upsert_customer_peer_recommendations(self, peer_recs: List[Dict[str, Any]]) -> bool:
        """Upsert recommendations derived from similar customer purchases."""
        if not self.client:
            logger.info(f"[Offline] Mock upsert {len(peer_recs)} peer-based collaborative recs.")
            return True
        try:
            self.client.table("customer_peer_recommendations").upsert(peer_recs).execute()
            return True
        except Exception as e:
            logger.error(f"Error upserting customer peer recommendations: {e}")
            return False

    def upsert_item_similarities(self, sim_records: List[Dict[str, Any]]) -> bool:
        """Upsert top-N item similarities."""
        if not self.client:
            logger.info(f"[Offline] Mock upsert {len(sim_records)} similarity pairs.")
            return True
        try:
            self.client.table("item_similarities").upsert(sim_records).execute()
            return True
        except Exception as e:
            logger.error(f"Error upserting item similarities: {e}")
            return False

    def upsert_customer_recommendations(self, rec_records: List[Dict[str, Any]]) -> bool:
        """Upsert personalized customer recommendations."""
        if not self.client:
            logger.info(f"[Offline] Mock upsert {len(rec_records)} customer recs.")
            return True
        try:
            self.client.table("customer_recommendations").upsert(rec_records).execute()
            return True
        except Exception as e:
            logger.error(f"Error upserting customer recommendations: {e}")
            return False

    def upsert_inventory_decisions(self, decisions: List[Dict[str, Any]]) -> bool:
        """Upsert ROP, safety stock, and price elasticity decisions."""
        if not self.client:
            logger.info(f"[Offline] Mock upsert {len(decisions)} inventory decisions.")
            return True
        try:
            self.client.table("inventory_pricing_decisions").upsert(decisions).execute()
            return True
        except Exception as e:
            logger.error(f"Error upserting inventory decisions: {e}")
            return False
