"""
Nightly Automated Sync Pipeline for Recosys Intelligence Hub
Executed every night via Windows Task Scheduler or cron.

Steps:
1. Ingests master catalog from BachirHome_2026 (Del=0, 2166 items).
2. Slices transactions across configured databases strictly for those items.
3. Computes 64-D Item2Vec & hybrid vectors, customer vectors, personalized recs, lookalikes.
4. Computes daily demand metrics, safety stock, ROP, price elasticity.
5. Computes monthly (Jan-Dec) and seasonal (Winter, Spring, Summer, Fall) forecasts.
6. Refreshes local web dashboard assets (web/data.js, web/full_data.json).
7. Upserts updated intelligence tables into Supabase (recosys_* tables).
8. Logs results to recosys_sync_logs and logs/nightly_sync.log.
"""

import sys
import os
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

from loguru import logger
logger.add(log_dir / "nightly_sync.log", rotation="20 MB", retention="30 days", level="INFO")

from export_3d_data import generate_production_intelligence_dataset
from scripts.sync_to_supabase import sync_json_to_supabase


def run_nightly_sync():
    start_ts = time.time()
    start_dt = datetime.now(timezone.utc)
    logger.info("=" * 70)
    logger.info(f"STARTING RECOSYS NIGHTLY SYNC at {start_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    logger.info("=" * 70)

    try:
        # Step 1: Run full pipeline on MSSQL and export web data
        logger.info("Step 1: Running MSSQL ingestion, ML modeling, and local export...")
        pipeline_stats = generate_production_intelligence_dataset()
        logger.info("Step 1 complete: Web data files refreshed.")

        # Step 2: Push dataset to Supabase
        json_path = str(PROJECT_ROOT / "web" / "full_data.json")
        logger.info(f"Step 2: Syncing dataset from {json_path} to Supabase...")
        sync_result = sync_json_to_supabase(json_path)
        logger.info(f"Step 2 complete: Supabase upsert finished. Result: {sync_result}")

        elapsed = time.time() - start_ts
        logger.info("=" * 70)
        logger.info(f"NIGHTLY SYNC COMPLETED SUCCESSFULLY in {elapsed:.1f}s ({elapsed/60:.2f} min)")
        logger.info(f"Synced {sync_result.get('items', 0)} items and {sync_result.get('customers', 0)} customers.")
        logger.info("=" * 70)
        return 0

    except Exception as e:
        elapsed = time.time() - start_ts
        logger.error("=" * 70)
        logger.error(f"NIGHTLY SYNC FAILED after {elapsed:.1f}s: {e}")
        logger.exception(e)
        logger.error("=" * 70)
        return 1


if __name__ == "__main__":
    exit_code = run_nightly_sync()
    sys.exit(exit_code)
