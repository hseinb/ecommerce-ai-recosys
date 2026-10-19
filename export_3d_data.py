import json
import os
import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
from datetime import datetime

from src.ingestion.mssql_client import MSSQLIngestionClient
from src.features.temporal import (
    add_temporal_cyclical_features,
    compute_monthly_demand_forecast,
    compute_seasonal_period_forecast,
)
from src.features.baskets import extract_transaction_baskets
from src.features.inventory_features import compute_daily_demand_metrics, compute_turnover_velocity
from src.embeddings.item2vec import Item2VecModel
from src.embeddings.hybrid import HybridEmbeddingBuilder
from src.embeddings.customer_embeddings import CustomerEmbeddingBuilder
from src.recommenders.user_personalized import UserPersonalizedRecommender
from src.recommenders.customer_lookalike import CustomerLookalikeRecommender
from src.inventory_pricing.safety_stock import SafetyStockCalculator
from src.inventory_pricing.elasticity import PriceElasticityEngine
from src.embeddings.gemini_service import GeminiService
from config.settings import settings


def generate_production_intelligence_dataset():
    logger.info("Ingesting real data from BachirHome_2024, BachirHome_2025, BachirHome_2026...")
    client = MSSQLIngestionClient()
    catalog_df = client.fetch_catalog()
    tx_df = client.fetch_transactions(days_back=None) # All days
    accounts_df = client.fetch_accounts()
    
    logger.info(f"Loaded {len(catalog_df)} catalog items, {len(accounts_df)} accounts, {len(tx_df)} transactions.")
    logger.info(f"Date Span: {tx_df['timestamp'].min()} to {tx_df['timestamp'].max()}")

    # 1. Temporal & Demand Aggregations
    tx_df_temporal = add_temporal_cyclical_features(tx_df, timestamp_col="timestamp")
    baskets = extract_transaction_baskets(tx_df_temporal)

    logger.info("Computing Monthly and Seasonal Demand Forecasts across 2024-2026...")
    monthly_forecast_df = compute_monthly_demand_forecast(tx_df_temporal)
    seasonal_period_df = compute_seasonal_period_forecast(tx_df_temporal)

    demand_metrics_df = compute_daily_demand_metrics(tx_df_temporal, date_col="date")
    inventory_state_df = compute_turnover_velocity(catalog_df, demand_metrics_df)

    # 2. Gemini AI Semantic Modeling & Item2Vec
    logger.info("Initializing Gemini Service for embeddings & sales intelligence...")
    gemini_service = GeminiService()

    logger.info("Fetching / generating Gemini semantic item embeddings (text-embedding-004)...")
    gemini_item_embeddings = gemini_service.get_item_embeddings(catalog_df)

    logger.info("Fetching / generating Gemini B2B sales pitches & merchandising notes (gemini-2.5-flash)...")
    gemini_pitches = gemini_service.get_item_sales_pitches(catalog_df)

    logger.info("Fetching / generating Gemini customer buyer personas & sales notes (gemini-2.5-flash)...")
    gemini_personas = gemini_service.get_customer_personas(accounts_df, tx_df_temporal)

    logger.info("Training Item2Vec model on transaction co-occurrence baskets...")
    item2vec = Item2VecModel(vector_size=32, epochs=settings.ITEM2VEC_EPOCHS)
    item2vec.fit(baskets)
    i2v_embeddings = item2vec.get_all_embeddings()

    logger.info("Building 64-dim Hybrid Item Embeddings (Gemini Semantics + Item2Vec)...")
    hybrid_builder = HybridEmbeddingBuilder(target_dim=settings.EMBEDDING_DIM)
    hybrid_embeddings = hybrid_builder.build_hybrid_vectors(
        items_df=catalog_df,
        item2vec_dict=i2v_embeddings,
        gemini_embeddings_dict=gemini_item_embeddings,
    )

    # 3. Customer Embeddings
    logger.info("Building 64-dim Customer Embeddings...")
    cust_builder = CustomerEmbeddingBuilder(half_life_days=settings.TIME_DECAY_HALF_LIFE_DAYS)
    customer_embeddings = cust_builder.build_all_customer_embeddings(
        transactions_df=tx_df_temporal,
        item_embeddings=hybrid_embeddings,
    )

    item_ids = list(hybrid_embeddings.keys())
    customer_ids = list(customer_embeddings.keys())
    
    logger.info(f"Active entities: {len(item_ids)} items, {len(customer_ids)} customer vectors.")

    # 4. Recommendation Engines & Lookalikes
    stock_map = {str(row["item_id"]): row["current_stock"] for row in catalog_df.iter_rows(named=True)}
    item_lookup = {str(row["item_id"]): row for row in catalog_df.iter_rows(named=True)}
    cust_lookup = {str(row["customer_id"]): row for row in accounts_df.iter_rows(named=True)}

    lookalike_engine = CustomerLookalikeRecommender(customer_embeddings)
    user_recommender = UserPersonalizedRecommender(hybrid_embeddings, half_life_days=settings.TIME_DECAY_HALF_LIFE_DAYS)

    customer_lookalikes_map = {}
    customer_peer_recs_map = {}
    customer_direct_recs_map = {}

    for c_id in customer_ids:
        # Top 5 lookalike peers
        peers = lookalike_engine.find_lookalike_customers(c_id, top_k=5)
        enriched_peers = []
        for p in peers:
            pid = str(p["lookalike_customer_id"])
            p_acc = cust_lookup.get(pid, {})
            enriched_peers.append({
                "customer_id": pid,
                "name": p_acc.get("customer_name", f"Account #{pid}"),
                "account_number": p_acc.get("account_number", ""),
                "similarity": round(float(p["similarity_score"]), 4),
            })
        customer_lookalikes_map[c_id] = enriched_peers

        # Peer recommendations
        p_recs = lookalike_engine.recommend_from_similar_customers(
            c_id, transactions_df=tx_df_temporal, top_k_items=8, n_lookalikes=5, stock_map=stock_map
        )
        enriched_precs = []
        for r in p_recs:
            iid = str(r["item_id"])
            i_info = item_lookup.get(iid, {})
            via_names = [cust_lookup.get(str(x), {}).get("customer_name", str(x)) for x in r.get("recommended_via_lookalike_ids", [])]
            pitch_info = gemini_pitches.get(iid, {})
            enriched_precs.append({
                "item_id": iid,
                "code": i_info.get("code", ""),
                "name": i_info.get("name", ""),
                "category": i_info.get("category_name", ""),
                "price": float(i_info.get("retail_price", 0.0)),
                "stock": int(i_info.get("current_stock", 0)),
                "score": round(float(r["collaborative_score"]), 4),
                "pitch": pitch_info.get("sales_pitch") or f"Recommended companion SKU in {i_info.get('category_name', 'Tableware')}.",
                "via": ", ".join(via_names),
            })
        customer_peer_recs_map[c_id] = enriched_precs

        # Direct personalized recs
        cust_tx = tx_df_temporal.filter(pl.col("customer_id") == c_id)
        d_recs = user_recommender.recommend_for_user(c_id, cust_tx, top_k=8, stock_map=stock_map)
        enriched_drecs = []
        for r in d_recs:
            iid = str(r["item_id"])
            i_info = item_lookup.get(iid, {})
            pitch_info = gemini_pitches.get(iid, {})
            enriched_drecs.append({
                "item_id": iid,
                "code": i_info.get("code", ""),
                "name": i_info.get("name", ""),
                "category": i_info.get("category_name", ""),
                "price": float(i_info.get("retail_price", 0.0)),
                "stock": int(i_info.get("current_stock", 0)),
                "score": round(float(r["predicted_score"]), 4),
                "pitch": pitch_info.get("sales_pitch") or f"Recommended high-affinity SKU in {i_info.get('category_name', 'Tableware')}.",
            })
        customer_direct_recs_map[c_id] = enriched_drecs

    # 5. Inventory & ROP Decisions
    logger.info("Computing inventory safety stock & price elasticity...")
    elasticity_engine = PriceElasticityEngine(n_clusters=5)
    cluster_map = elasticity_engine.cluster_items(hybrid_embeddings)
    cluster_ped_results = elasticity_engine.estimate_cluster_elasticity(tx_df_temporal, cluster_map=cluster_map)

    stock_calc = SafetyStockCalculator(z_score=settings.DEFAULT_SERVICE_LEVEL_Z)
    inventory_decisions = []
    for item in inventory_state_df.iter_rows(named=True):
        i_id = str(item["item_id"])
        c_id = cluster_map.get(i_id, 0)
        ped_info = cluster_ped_results.get(c_id, {"ped": -1.0, "recommendation": "MAINTAIN"})
        
        rop_res = stock_calc.calculate_item_rop(
            avg_daily_demand=item.get("avg_daily_demand") or 0.0,
            std_daily_demand=item.get("std_daily_demand") or 0.0,
            lead_time_days=item.get("supplier_lead_time_days") or 7.0,
            current_stock=item.get("current_stock") or 0,
        )

        pitch_info = gemini_pitches.get(i_id, {})
        inventory_decisions.append({
            "item_id": i_id,
            "code": item.get("code", ""),
            "name": item.get("name", ""),
            "category_name": item.get("category_name", ""),
            "current_stock": int(item.get("current_stock", 0)),
            "retail_price": float(item.get("retail_price", 0.0)),
            "unit_cost": float(item.get("unit_cost", 0.0)),
            "margin": round(float(item.get("margin", 0.0)), 3),
            "avg_daily_demand": round(float(rop_res["avg_daily_demand"]), 3),
            "safety_stock": round(float(rop_res["safety_stock"]), 1),
            "reorder_point": round(float(rop_res["reorder_point"]), 1),
            "stock_status": rop_res["stock_status"],
            "elasticity_ped": round(float(ped_info["ped"]), 3),
            "price_action": ped_info["recommendation"],
            "sales_pitch": pitch_info.get("sales_pitch", ""),
        })

    # 6. Joint 3D Manifold Embedding Projection (Items + Customers)
    logger.info("Computing joint 3D manifold for items and customers...")
    item_matrix = np.array([hybrid_embeddings[i_id] for i_id in item_ids], dtype=np.float32)
    cust_matrix = np.array([customer_embeddings[c_id] for c_id in customer_ids], dtype=np.float32)
    combined_matrix = np.vstack([item_matrix, cust_matrix])

    # K-Means clustering on items
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    item_cluster_labels = kmeans.fit_predict(item_matrix)

    # Group items by cluster for Gemini merchandising collection naming
    cluster_items_map = {}
    for idx, i_id in enumerate(item_ids):
        c_id = int(item_cluster_labels[idx])
        if c_id not in cluster_items_map:
            cluster_items_map[c_id] = []
        cluster_items_map[c_id].append(item_lookup.get(str(i_id), {}))

    logger.info("Generating Gemini curated merchandising names for 3D clusters (gemini-3.8-flash)...")
    cluster_names = gemini_service.generate_cluster_names(cluster_items_map)

    # PCA 3D
    pca = PCA(n_components=3, random_state=42)
    combined_pca = pca.fit_transform(combined_matrix)
    combined_pca = (combined_pca - combined_pca.mean(axis=0)) / (combined_pca.std(axis=0) + 1e-6) * 40.0

    # t-SNE 3D
    tsne = TSNE(n_components=3, perplexity=30, random_state=42, max_iter=1000)
    combined_tsne = tsne.fit_transform(combined_matrix)
    combined_tsne = (combined_tsne - combined_tsne.mean(axis=0)) / (combined_tsne.std(axis=0) + 1e-6) * 40.0

    # Item nearest neighbors via cosine similarity
    item_sim_matrix = cosine_similarity(item_matrix)

    cluster_color_palette = [
        "#00f0ff", # Cyan
        "#ff007f", # Magenta
        "#8b5cf6", # Violet
        "#00ff88", # Emerald
        "#ffaa00", # Amber
        "#ff3366", # Crimson
        "#00b4d8", # Sky Blue
        "#f72585", # Rose
    ]

    items_payload = []
    n_items = len(item_ids)

    for idx, i_id in enumerate(item_ids):
        row = item_lookup.get(str(i_id), {})
        row_sims = item_sim_matrix[idx]
        sorted_indices = np.argsort(-row_sims)
        neighbors = []
        for n_idx in sorted_indices:
            if n_idx == idx:
                continue
            n_id = item_ids[n_idx]
            n_row = item_lookup.get(str(n_id), {})
            neighbors.append({
                "id": str(n_id),
                "index": int(n_idx),
                "code": str(n_row.get("code", "")),
                "name": str(n_row.get("name", "")),
                "category": str(n_row.get("category_name", "")),
                "price": float(n_row.get("retail_price", 0.0)),
                "similarity": round(float(row_sims[n_idx]), 4),
            })
            if len(neighbors) >= 5:
                break

        c_id = int(item_cluster_labels[idx])
        c_name = cluster_names.get(c_id, f"Collection {c_id + 1}")
        pitch_info = gemini_pitches.get(str(i_id), {})

        items_payload.append({
            "type": "item",
            "id": str(i_id),
            "index": idx,
            "code": str(row.get("code", "")),
            "name": str(row.get("name", "")),
            "category": str(row.get("category_name", "General")),
            "price": float(row.get("retail_price", 0.0)),
            "cost": float(row.get("unit_cost", 0.0)),
            "stock": int(row.get("current_stock", 0)),
            "margin": round(float(row.get("margin", 0.0)), 3),
            "cluster_id": c_id,
            "cluster_name": c_name,
            "sales_pitch": pitch_info.get("sales_pitch", ""),
            "merchandising_notes": pitch_info.get("merchandising_notes", ""),
            "color": cluster_color_palette[c_id % len(cluster_color_palette)],
            "pca": {
                "x": round(float(combined_pca[idx, 0]), 3),
                "y": round(float(combined_pca[idx, 1]), 3),
                "z": round(float(combined_pca[idx, 2]), 3),
            },
            "tsne": {
                "x": round(float(combined_tsne[idx, 0]), 3),
                "y": round(float(combined_tsne[idx, 1]), 3),
                "z": round(float(combined_tsne[idx, 2]), 3),
            },
            "neighbors": neighbors,
        })

    # Customers 3D payload
    customers_payload = []
    for c_idx, c_id in enumerate(customer_ids):
        global_idx = n_items + c_idx
        c_acc = cust_lookup.get(str(c_id), {})
        persona_info = gemini_personas.get(str(c_id), {})
        
        customers_payload.append({
            "type": "customer",
            "id": str(c_id),
            "index": global_idx,
            "name": str(c_acc.get("customer_name", f"Customer {c_id}")),
            "account_number": str(c_acc.get("account_number", "")),
            "balance": float(c_acc.get("balance", 0.0)),
            "buyer_persona": persona_info.get("buyer_persona", ""),
            "sales_notes": persona_info.get("sales_notes", ""),
            "color": "#ffd700", # Radiant Gold
            "pca": {
                "x": round(float(combined_pca[global_idx, 0]), 3),
                "y": round(float(combined_pca[global_idx, 1]), 3),
                "z": round(float(combined_pca[global_idx, 2]), 3),
            },
            "tsne": {
                "x": round(float(combined_tsne[global_idx, 0]), 3),
                "y": round(float(combined_tsne[global_idx, 1]), 3),
                "z": round(float(combined_tsne[global_idx, 2]), 3),
            },
            "lookalikes": customer_lookalikes_map.get(c_id, []),
            "personalized_recs": customer_direct_recs_map.get(c_id, []),
            "peer_recs": customer_peer_recs_map.get(c_id, []),
        })

    # 7. Format Monthly Demand Forecast (all 12 months)
    month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    monthly_forecast_list = []
    for row in monthly_forecast_df.iter_rows(named=True):
        iid = str(row["item_id"])
        i_info = item_lookup.get(iid, {})
        m_num = int(row["month"])
        monthly_forecast_list.append({
            "item_id": iid,
            "code": i_info.get("code", ""),
            "name": i_info.get("name", ""),
            "category": i_info.get("category_name", ""),
            "price": float(i_info.get("retail_price", 0.0)),
            "current_stock": int(i_info.get("current_stock", 0)),
            "month_number": m_num,
            "month_name": month_names[m_num - 1] if 1 <= m_num <= 12 else f"Month {m_num}",
            "month_total_qty": round(float(row["month_total_qty"]), 1),
            "monthly_velocity_index": round(float(row["monthly_velocity_index"]), 2),
            "demand_tier": row["demand_tier"],
        })

    # 8. Format Seasonal Period Forecast (Winter, Spring, Summer, Fall)
    seasonal_period_list = []
    for row in seasonal_period_df.iter_rows(named=True):
        iid = str(row["item_id"])
        i_info = item_lookup.get(iid, {})
        seasonal_period_list.append({
            "item_id": iid,
            "code": i_info.get("code", ""),
            "name": i_info.get("name", ""),
            "category": i_info.get("category_name", ""),
            "price": float(i_info.get("retail_price", 0.0)),
            "current_stock": int(i_info.get("current_stock", 0)),
            "season": row["season"],
            "season_total_qty": round(float(row["season_total_qty"]), 1),
            "seasonal_velocity_index": round(float(row["seasonal_velocity_index"]), 2),
            "season_tier": row["season_tier"],
        })

    full_payload = {
        "items": items_payload,
        "customers": customers_payload,
        "inventory": inventory_decisions,
        "seasonal_monthly": monthly_forecast_list,
        "seasonal_period": seasonal_period_list,
        "summary": {
            "total_items": len(items_payload),
            "total_customers": len(customers_payload),
            "total_transactions": len(tx_df),
            "databases": settings.database_list,
            "date_min": str(tx_df["timestamp"].min()),
            "date_max": str(tx_df["timestamp"].max()),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    }

    os.makedirs("web", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Save to outputs CSVs
    pl.DataFrame(monthly_forecast_list).write_csv("outputs/seasonal_forecast_monthly.csv")
    pl.DataFrame(seasonal_period_list).write_csv("outputs/seasonal_forecast_by_season.csv")
    pl.DataFrame(inventory_decisions).write_csv("outputs/inventory_decisions.csv")

    # Save to web
    js_output_path = "web/data.js"
    json_output_path = "web/full_data.json"

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, ensure_ascii=False)

    with open(js_output_path, "w", encoding="utf-8") as f:
        f.write("window.RECOSYS_DATA = ")
        json.dump(full_payload, f, ensure_ascii=False)
        f.write(";\n")

    logger.info(f"Successfully exported full multi-year production dataset to {js_output_path} and {json_output_path}!")


if __name__ == "__main__":
    generate_production_intelligence_dataset()
