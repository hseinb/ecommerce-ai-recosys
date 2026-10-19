from datetime import datetime
from loguru import logger
import polars as pl

from src.ingestion.mssql_client import MSSQLIngestionClient
from src.features.temporal import add_temporal_cyclical_features, compute_seasonal_velocity_indices
from src.features.baskets import extract_transaction_baskets
from src.features.inventory_features import compute_daily_demand_metrics, compute_turnover_velocity
from src.embeddings.item2vec import Item2VecModel
from src.embeddings.hybrid import HybridEmbeddingBuilder
from src.embeddings.customer_embeddings import CustomerEmbeddingBuilder
from src.recommenders.item_to_item import ItemToItemRecommender
from src.recommenders.user_personalized import UserPersonalizedRecommender
from src.recommenders.customer_lookalike import CustomerLookalikeRecommender
from src.recommenders.seasonal_boosting import SeasonalBoostingEngine
from src.inventory_pricing.safety_stock import SafetyStockCalculator
from src.inventory_pricing.elasticity import PriceElasticityEngine
from src.database.supabase_client import SupabaseManager
from config.settings import settings


def run_full_recosys_pipeline():
    logger.info("================ STARTING RECOSYS & INVENTORY PIPELINE ================")
    
    # -------------------------------------------------------------------------
    # Pipeline 1: Ingestion & Feature Engineering
    # -------------------------------------------------------------------------
    logger.info("[Pipeline 1] Ingesting transactions & catalog...")
    ingest_client = MSSQLIngestionClient()
    catalog_df = ingest_client.fetch_catalog()
    tx_df = ingest_client.fetch_transactions(days_back=100)
    accounts_df = ingest_client.fetch_accounts()
    
    logger.info(f"Loaded {len(catalog_df)} catalog items, {len(accounts_df)} customer accounts, and {len(tx_df)} transaction line items.")

    # Temporal feature engineering
    tx_df_temporal = add_temporal_cyclical_features(tx_df, timestamp_col="timestamp")
    seasonal_indices_df = compute_seasonal_velocity_indices(tx_df_temporal)
    
    # Inventory & demand metrics
    demand_metrics_df = compute_daily_demand_metrics(tx_df_temporal, date_col="date")
    inventory_state_df = compute_turnover_velocity(catalog_df, demand_metrics_df)

    # -------------------------------------------------------------------------
    # Pipeline 2: Spatial Representation (Item & Customer Embeddings)
    # -------------------------------------------------------------------------
    logger.info("[Pipeline 2] Extracting transaction baskets & training Item2Vec...")
    baskets = extract_transaction_baskets(tx_df_temporal)
    item2vec = Item2VecModel(vector_size=32, epochs=settings.ITEM2VEC_EPOCHS)
    item2vec.fit(baskets)
    i2v_embeddings = item2vec.get_all_embeddings()

    logger.info("[Pipeline 2] Constructing 64-dim Hybrid Latent Item Embeddings...")
    hybrid_builder = HybridEmbeddingBuilder(target_dim=settings.EMBEDDING_DIM)
    hybrid_embeddings = hybrid_builder.build_hybrid_vectors(
        items_df=catalog_df,
        item2vec_dict=i2v_embeddings,
    )

    logger.info("[Pipeline 2] Constructing 64-dim Customer Embeddings in shared latent space...")
    cust_builder = CustomerEmbeddingBuilder(half_life_days=settings.TIME_DECAY_HALF_LIFE_DAYS)
    customer_embeddings = cust_builder.build_all_customer_embeddings(
        transactions_df=tx_df_temporal,
        item_embeddings=hybrid_embeddings,
    )

    # -------------------------------------------------------------------------
    # Pipeline 3: Recommendation Engines
    # -------------------------------------------------------------------------
    logger.info("[Pipeline 3] Generating Item-to-Item, User Direct, and Customer-to-Customer Peer Recs...")
    stock_map = {row["item_id"]: row["current_stock"] for row in catalog_df.iter_rows(named=True)}
    category_map = {row["item_id"]: row["category_name"] for row in catalog_df.iter_rows(named=True)}

    # A. Item-to-Item (Item Cosine Distance)
    i2i_recommender = ItemToItemRecommender(hybrid_embeddings)
    all_item_similarities = i2i_recommender.batch_generate_similarities(top_k=settings.ANN_TOP_K, stock_map=stock_map)

    # B. Customer Lookalikes & Peer Recommendations ("One purchase from similar customer -> recommended")
    lookalike_engine = CustomerLookalikeRecommender(customer_embeddings)
    active_customers = list(customer_embeddings.keys())
    
    all_lookalike_pairs = []
    peer_collaborative_recs = []

    for cust_id in active_customers:
        # Top 5 lookalike peers
        peers = lookalike_engine.find_lookalike_customers(cust_id, top_k=5)
        all_lookalike_pairs.extend(peers)

        # Peer purchase recommendations
        peer_recs = lookalike_engine.recommend_from_similar_customers(
            cust_id,
            transactions_df=tx_df_temporal,
            top_k_items=10,
            n_lookalikes=5,
            stock_map=stock_map,
        )
        peer_collaborative_recs.extend(peer_recs)

    # C. Direct User Personalized Recs (Time-decay user vector vs catalog)
    user_recommender = UserPersonalizedRecommender(
        hybrid_embeddings,
        half_life_days=settings.TIME_DECAY_HALF_LIFE_DAYS,
    )
    customer_direct_recs = []
    for cust_id in active_customers:
        cust_tx = tx_df_temporal.filter(pl.col("customer_id") == cust_id)
        recs = user_recommender.recommend_for_user(cust_id, cust_tx, top_k=10, stock_map=stock_map)
        customer_direct_recs.extend(recs)

    # D. Seasonal Boosting & Featured Items
    seasonal_engine = SeasonalBoostingEngine(alpha=settings.SEASONAL_BOOST_ALPHA)
    current_week = datetime.now().isocalendar()[1]
    current_quarter = (datetime.now().month - 1) // 3 + 1
    featured_items = seasonal_engine.get_featured_items_for_period(
        seasonal_indices_df,
        week=current_week,
        quarter=current_quarter,
        top_k=20,
    )

    # -------------------------------------------------------------------------
    # Pipeline 4: Inventory & Pricing Decision Layer
    # -------------------------------------------------------------------------
    logger.info("[Pipeline 4] Calculating Safety Stock, ROP, and Price Elasticity...")
    
    # Clustering & Elasticity
    elasticity_engine = PriceElasticityEngine(n_clusters=5)
    cluster_map = elasticity_engine.cluster_items(hybrid_embeddings)
    cluster_ped_results = elasticity_engine.estimate_cluster_elasticity(
        tx_df_temporal,
        cluster_map=cluster_map,
    )

    # Safety Stock & Reorder Points
    stock_calc = SafetyStockCalculator(z_score=settings.DEFAULT_SERVICE_LEVEL_Z)
    inventory_decisions = []
    
    for item in inventory_state_df.iter_rows(named=True):
        i_id = item["item_id"]
        c_id = cluster_map.get(i_id, 0)
        ped_info = cluster_ped_results.get(c_id, {"ped": -1.0, "recommendation": "MAINTAIN"})
        
        rop_res = stock_calc.calculate_item_rop(
            avg_daily_demand=item.get("avg_daily_demand") or 0.0,
            std_daily_demand=item.get("std_daily_demand") or 0.0,
            lead_time_days=item.get("supplier_lead_time_days") or 7.0,
            current_stock=item.get("current_stock") or 0,
        )

        inventory_decisions.append({
            "item_id": i_id,
            "code": item.get("code", ""),
            "name": item.get("name", ""),
            "category_name": item.get("category_name", ""),
            "cluster_id": c_id,
            "current_stock": item.get("current_stock", 0),
            "retail_price": item.get("retail_price", 0.0),
            "unit_cost": item.get("unit_cost", 0.0),
            "margin": item.get("margin", 0.0),
            "avg_daily_demand": round(rop_res["avg_daily_demand"], 3),
            "demand_std_dev": round(rop_res["std_daily_demand"], 3),
            "lead_time_days": rop_res["lead_time_days"],
            "safety_stock": rop_res["safety_stock"],
            "reorder_point": rop_res["reorder_point"],
            "stock_status": rop_res["stock_status"],
            "price_elasticity_ped": round(ped_info["ped"], 3),
            "recommended_price_action": ped_info["recommendation"],
        })

    # -------------------------------------------------------------------------
    # Pipeline Output Export: Save result tables to outputs/ directory & SQLite
    # -------------------------------------------------------------------------
    logger.info("[Outputs] Exporting result tables to outputs/ CSV files & SQLite...")
    import os
    import sqlite3
    os.makedirs("outputs", exist_ok=True)

    item_lookup = {str(row["item_id"]): row for row in catalog_df.iter_rows(named=True)}
    cust_lookup = {str(row["customer_id"]): row for row in accounts_df.iter_rows(named=True)}

    # 1. Enriched Direct Customer Recommendations
    enriched_customer_recs = []
    for r in customer_direct_recs:
        c_id = str(r.get("customer_id"))
        i_id = str(r.get("item_id"))
        c_info = cust_lookup.get(c_id, {})
        i_info = item_lookup.get(i_id, {})
        enriched_customer_recs.append({
            "customer_id": c_id,
            "customer_name": c_info.get("customer_name", "Unknown"),
            "account_number": c_info.get("account_number", ""),
            "item_id": i_id,
            "item_code": i_info.get("code", ""),
            "item_name": i_info.get("name", ""),
            "category_name": i_info.get("category_name", ""),
            "retail_price": i_info.get("retail_price", 0.0),
            "current_stock": i_info.get("current_stock", 0),
            "predicted_score": round(float(r.get("predicted_score", 0.0)), 4),
            "rank": r.get("rank", 1),
        })
    df_recs = pl.DataFrame(enriched_customer_recs) if enriched_customer_recs else pl.DataFrame()
    if not df_recs.is_empty():
        df_recs.write_csv("outputs/recommendations_personalized.csv")

    # 2. Enriched Peer Collaborative Recommendations
    enriched_peer_recs = []
    for r in peer_collaborative_recs:
        c_id = str(r.get("customer_id"))
        i_id = str(r.get("item_id"))
        via_ids = [str(x) for x in r.get("recommended_via_lookalike_ids", [])]
        via_names = [cust_lookup.get(vid, {}).get("customer_name", vid) for vid in via_ids]

        c_info = cust_lookup.get(c_id, {})
        i_info = item_lookup.get(i_id, {})
        enriched_peer_recs.append({
            "customer_id": c_id,
            "customer_name": c_info.get("customer_name", "Unknown"),
            "account_number": c_info.get("account_number", ""),
            "item_id": i_id,
            "item_code": i_info.get("code", ""),
            "item_name": i_info.get("name", ""),
            "category_name": i_info.get("category_name", ""),
            "retail_price": i_info.get("retail_price", 0.0),
            "current_stock": i_info.get("current_stock", 0),
            "recommended_via_lookalike_customers": ", ".join(via_names),
            "collaborative_score": round(float(r.get("collaborative_score", 0.0)), 4),
            "rank": r.get("rank", 1),
        })
    df_peer_recs = pl.DataFrame(enriched_peer_recs) if enriched_peer_recs else pl.DataFrame()
    if not df_peer_recs.is_empty():
        df_peer_recs.write_csv("outputs/recommendations_peer_collaborative.csv")

    # 3. Enriched Customer Lookalikes
    enriched_lookalikes = []
    for r in all_lookalike_pairs:
        c1_id = str(r.get("customer_id"))
        c2_id = str(r.get("lookalike_customer_id"))
        c1 = cust_lookup.get(c1_id, {})
        c2 = cust_lookup.get(c2_id, {})
        enriched_lookalikes.append({
            "customer_id": c1_id,
            "customer_name": c1.get("customer_name", "Unknown"),
            "account_number": c1.get("account_number", ""),
            "similar_customer_id": c2_id,
            "similar_customer_name": c2.get("customer_name", "Unknown"),
            "similar_account_number": c2.get("account_number", ""),
            "similarity_score": round(float(r.get("similarity_score", 0.0)), 4),
            "rank": r.get("rank", 1),
        })
    df_lookalikes = pl.DataFrame(enriched_lookalikes) if enriched_lookalikes else pl.DataFrame()
    if not df_lookalikes.is_empty():
        df_lookalikes.write_csv("outputs/customer_lookalikes.csv")

    # 4. Enriched Item-to-Item Similarities
    enriched_item_sims = []
    for r in all_item_similarities:
        i1_id = str(r.get("source_item_id"))
        i2_id = str(r.get("target_item_id"))
        i1 = item_lookup.get(i1_id, {})
        i2 = item_lookup.get(i2_id, {})
        enriched_item_sims.append({
            "source_item_id": i1_id,
            "source_code": i1.get("code", ""),
            "source_name": i1.get("name", ""),
            "source_category": i1.get("category_name", ""),
            "target_item_id": i2_id,
            "target_code": i2.get("code", ""),
            "target_name": i2.get("name", ""),
            "target_category": i2.get("category_name", ""),
            "target_price": i2.get("retail_price", 0.0),
            "target_stock": i2.get("current_stock", 0),
            "similarity_score": round(float(r.get("similarity_score", 0.0)), 4),
            "rank": r.get("rank", 1),
        })
    df_item_sims = pl.DataFrame(enriched_item_sims) if enriched_item_sims else pl.DataFrame()
    if not df_item_sims.is_empty():
        df_item_sims.write_csv("outputs/item_similarities.csv")

    # 5. Inventory & Reorder Decisions
    df_inventory = pl.DataFrame(inventory_decisions)
    df_inventory.write_csv("outputs/inventory_decisions.csv")

    # 6. Seasonal Featured Items
    enriched_featured = []
    for r in featured_items:
        i_id = str(r.get("item_id"))
        i_info = item_lookup.get(i_id, {})
        enriched_featured.append({
            "item_id": i_id,
            "item_code": i_info.get("code", ""),
            "item_name": i_info.get("name", ""),
            "category_name": i_info.get("category_name", ""),
            "retail_price": i_info.get("retail_price", 0.0),
            "current_stock": i_info.get("current_stock", 0),
            "seasonal_velocity_index": round(float(r.get("seasonal_velocity_index", 1.0)), 3),
        })
    df_featured = pl.DataFrame(enriched_featured) if enriched_featured else pl.DataFrame()
    if not df_featured.is_empty():
        df_featured.write_csv("outputs/seasonal_featured_items.csv")

    # Save all tables to SQLite database for instant SQL querying
    sqlite_db_path = "outputs/recosys_results.db"
    conn = sqlite3.connect(sqlite_db_path)
    if not df_recs.is_empty():
        df_recs.to_pandas().to_sql("recommendations_personalized", conn, if_exists="replace", index=False)
    if not df_peer_recs.is_empty():
        df_peer_recs.to_pandas().to_sql("recommendations_peer_collaborative", conn, if_exists="replace", index=False)
    if not df_lookalikes.is_empty():
        df_lookalikes.to_pandas().to_sql("customer_lookalikes", conn, if_exists="replace", index=False)
    if not df_item_sims.is_empty():
        df_item_sims.to_pandas().to_sql("item_similarities", conn, if_exists="replace", index=False)
    if not df_inventory.is_empty():
        df_inventory.to_pandas().to_sql("inventory_decisions", conn, if_exists="replace", index=False)
    if not df_featured.is_empty():
        df_featured.to_pandas().to_sql("seasonal_featured_items", conn, if_exists="replace", index=False)
    conn.close()
    logger.info(f"[Outputs] Saved all tables to CSV and SQLite database at {sqlite_db_path}")

    # -------------------------------------------------------------------------
    # Pipeline Storage: Supabase & pgvector Upsert (Optional)
    # -------------------------------------------------------------------------
    logger.info("[Database] Upserting pipeline results to Supabase...")
    supabase = SupabaseManager()
    
    item_embeddings_payload = [
        {"item_id": i_id, "embedding": vec.tolist(), "cluster_id": cluster_map.get(i_id, 0)}
        for i_id, vec in hybrid_embeddings.items()
    ]

    customer_embeddings_payload = [
        {"customer_id": c_id, "embedding": vec.tolist(), "persona_segment_id": 0}
        for c_id, vec in customer_embeddings.items()
    ]
    
    supabase.upsert_items(catalog_df.to_dicts())
    supabase.upsert_item_embeddings(item_embeddings_payload)
    supabase.upsert_customer_embeddings(customer_embeddings_payload)
    supabase.upsert_item_similarities(all_item_similarities[:100])
    supabase.upsert_customer_lookalikes(all_lookalike_pairs)
    supabase.upsert_customer_peer_recommendations(peer_collaborative_recs)
    supabase.upsert_customer_recommendations(customer_direct_recs)
    supabase.upsert_inventory_decisions(inventory_decisions)

    logger.info("================ PIPELINE RUN COMPLETE ================")
    logger.info(
        f"Summary: {len(hybrid_embeddings)} item embeddings, {len(customer_embeddings)} customer embeddings, "
        f"{len(all_lookalike_pairs)} lookalike pairs, {len(peer_collaborative_recs)} peer recs, "
        f"{len(inventory_decisions)} inventory decisions exported to outputs/."
    )
    return {
        "item_embeddings_count": len(hybrid_embeddings),
        "customer_embeddings_count": len(customer_embeddings),
        "lookalike_pairs_count": len(all_lookalike_pairs),
        "peer_recs_count": len(peer_collaborative_recs),
        "inventory_decisions_count": len(inventory_decisions),
    }


if __name__ == "__main__":
    run_full_recosys_pipeline()
