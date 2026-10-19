"""
Sync Recosys Intelligence Dataset to Supabase
Upserts items, customers, inventory ROP, monthly forecasts, and seasonal forecasts
into dedicated isolated tables (recosys_*) in Supabase.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from supabase import create_client, Client
from config.settings import settings

# Setup logging
log_dir = Path(__file__).resolve().parent.parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
logger.add(log_dir / "supabase_sync.log", rotation="10 MB", retention="14 days")


def get_supabase_client() -> Client:
    if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be configured in .env")
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)


def batch_upsert(client: Client, table_name: str, records: list, batch_size: int = 400, on_conflict: str = ""):
    total = len(records)
    logger.info(f"Upserting {total} records into '{table_name}' in batches of {batch_size}...")
    for i in range(0, total, batch_size):
        batch = records[i:i + batch_size]
        try:
            if on_conflict:
                client.table(table_name).upsert(batch, on_conflict=on_conflict).execute()
            else:
                client.table(table_name).upsert(batch).execute()
        except Exception as e:
            logger.error(f"Error upserting batch {i//batch_size + 1} into {table_name}: {e}")
            raise e
    logger.info(f"Successfully upserted {total} records into '{table_name}'.")


def sync_json_to_supabase(json_path: str = "web/full_data.json") -> dict:
    start_time = datetime.now(timezone.utc)
    logger.info(f"Starting Supabase sync from {json_path} at {start_time.isoformat()}...")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Data file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    client = get_supabase_client()

    # Log start in recosys_sync_logs
    sync_log_entry = {
        "sync_started_at": start_time.isoformat(),
        "status": "RUNNING",
        "items_synced": 0,
        "customers_synced": 0,
        "transactions_processed": 46835,
        "details": {"source": json_path}
    }
    log_res = client.table("recosys_sync_logs").insert(sync_log_entry).execute()
    log_id = log_res.data[0]["id"] if log_res.data else None

    try:
        # 1. Prepare Items
        raw_items = data.get("items", [])
        items_payload = []
        for it in raw_items:
            items_payload.append({
                "item_id": str(it.get("id")),
                "code": it.get("code"),
                "name": it.get("name"),
                "category": it.get("category"),
                "price": float(it.get("price", 0.0)),
                "cost": float(it.get("cost", 0.0)),
                "stock": float(it.get("stock", 0.0)),
                "cluster_id": int(it.get("cluster_id", 0)),
                "cluster_name": it.get("cluster_name"),
                "sales_pitch": it.get("sales_pitch"),
                "merchandising_notes": it.get("merchandising_notes"),
                "color": it.get("color"),
                "pca_x": float(it.get("pca", {}).get("x", 0.0)),
                "pca_y": float(it.get("pca", {}).get("y", 0.0)),
                "pca_z": float(it.get("pca", {}).get("z", 0.0)),
                "tsne_x": float(it.get("tsne", {}).get("x", 0.0)),
                "tsne_y": float(it.get("tsne", {}).get("y", 0.0)),
                "tsne_z": float(it.get("tsne", {}).get("z", 0.0)),
                "neighbors": it.get("neighbors", []),
                "updated_at": datetime.now(timezone.utc).isoformat()
            })
        if items_payload:
            batch_upsert(client, "recosys_items", items_payload, on_conflict="item_id")

        # 2. Prepare Customers
        raw_customers = data.get("customers", [])
        customers_payload = []
        for c in raw_customers:
            customers_payload.append({
                "customer_id": str(c.get("id")),
                "account_number": c.get("account_number"),
                "name": c.get("name"),
                "balance": float(c.get("balance", 0.0)),
                "buyer_persona": c.get("buyer_persona"),
                "sales_notes": c.get("sales_notes"),
                "pca_x": float(c.get("pca", {}).get("x", 0.0)),
                "pca_y": float(c.get("pca", {}).get("y", 0.0)),
                "pca_z": float(c.get("pca", {}).get("z", 0.0)),
                "tsne_x": float(c.get("tsne", {}).get("x", 0.0)),
                "tsne_y": float(c.get("tsne", {}).get("y", 0.0)),
                "tsne_z": float(c.get("tsne", {}).get("z", 0.0)),
                "lookalikes": c.get("lookalikes", []),
                "personalized_recs": c.get("personalized_recs", []),
                "peer_recs": c.get("peer_recs", []),
                "updated_at": datetime.now(timezone.utc).isoformat()
            })
        if customers_payload:
            batch_upsert(client, "recosys_customers", customers_payload, on_conflict="customer_id")

        # 3. Prepare Inventory ROP
        raw_inv = data.get("inventory", [])
        inv_payload = []
        for inv in raw_inv:
            inv_payload.append({
                "item_id": str(inv.get("item_id")),
                "code": inv.get("code"),
                "name": inv.get("name"),
                "category": inv.get("category_name"),
                "current_stock": float(inv.get("current_stock", 0.0)),
                "avg_daily_demand": float(inv.get("avg_daily_demand", 0.0)),
                "safety_stock": float(inv.get("safety_stock", 0.0)),
                "reorder_point": float(inv.get("reorder_point", 0.0)),
                "stock_status": inv.get("stock_status"),
                "retail_price": float(inv.get("retail_price", 0.0)),
                "price_action": inv.get("price_action"),
                "sales_pitch": inv.get("sales_pitch"),
                "updated_at": datetime.now(timezone.utc).isoformat()
            })
        if inv_payload:
            batch_upsert(client, "recosys_inventory_rop", inv_payload, on_conflict="item_id")

        # 4. Prepare Monthly Seasonal Forecast
        raw_monthly = data.get("seasonal_monthly", [])
        monthly_payload = []
        for m in raw_monthly:
            monthly_payload.append({
                "item_id": str(m.get("item_id")),
                "month_number": int(m.get("month_number")),
                "code": m.get("code"),
                "name": m.get("name"),
                "category": m.get("category"),
                "price": float(m.get("price", 0.0)),
                "current_stock": float(m.get("current_stock", 0.0)),
                "month_name": m.get("month_name"),
                "month_total_qty": float(m.get("month_total_qty", 0.0)),
                "monthly_velocity_index": float(m.get("monthly_velocity_index", 1.0)),
                "demand_tier": m.get("demand_tier"),
                "updated_at": datetime.now(timezone.utc).isoformat()
            })
        if monthly_payload:
            batch_upsert(client, "recosys_seasonal_monthly", monthly_payload, on_conflict="item_id,month_number")

        # 5. Prepare Seasonal Period Forecast
        raw_period = data.get("seasonal_period", [])
        period_payload = []
        for s in raw_period:
            period_payload.append({
                "item_id": str(s.get("item_id")),
                "season": s.get("season"),
                "code": s.get("code"),
                "name": s.get("name"),
                "category": s.get("category"),
                "price": float(s.get("price", 0.0)),
                "current_stock": float(s.get("current_stock", 0.0)),
                "season_total_qty": float(s.get("season_total_qty", 0.0)),
                "seasonal_velocity_index": float(s.get("seasonal_velocity_index", 1.0)),
                "season_tier": s.get("season_tier"),
                "updated_at": datetime.now(timezone.utc).isoformat()
            })
        if period_payload:
            batch_upsert(client, "recosys_seasonal_period", period_payload, on_conflict="item_id,season")

        end_time = datetime.now(timezone.utc)
        duration_sec = (end_time - start_time).total_seconds()
        logger.info(f"Sync successfully completed in {duration_sec:.1f} seconds!")

        # Update log
        if log_id:
            client.table("recosys_sync_logs").update({
                "sync_finished_at": end_time.isoformat(),
                "status": "SUCCESS",
                "items_synced": len(items_payload),
                "customers_synced": len(customers_payload),
                "details": {
                    "duration_seconds": duration_sec,
                    "monthly_records": len(monthly_payload),
                    "seasonal_records": len(period_payload)
                }
            }).eq("id", log_id).execute()

        return {
            "status": "SUCCESS",
            "items": len(items_payload),
            "customers": len(customers_payload),
            "monthly": len(monthly_payload),
            "seasonal": len(period_payload),
            "duration_sec": duration_sec
        }

    except Exception as e:
        end_time = datetime.now(timezone.utc)
        logger.error(f"Sync failed: {e}")
        if log_id:
            client.table("recosys_sync_logs").update({
                "sync_finished_at": end_time.isoformat(),
                "status": "FAILED",
                "details": {"error": str(e)}
            }).eq("id", log_id).execute()
        raise e


if __name__ == "__main__":
    sync_json_to_supabase()
