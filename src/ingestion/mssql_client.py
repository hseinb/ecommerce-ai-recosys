from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import numpy as np
import polars as pl
from loguru import logger

try:
    import pyodbc
    import sqlalchemy as sa
except ImportError:
    pyodbc = None
    sa = None

from config.settings import settings


class MSSQLIngestionClient:
    """
    Connects to legacy MSSQL tables (Compatibility Level 100 / SQL Server 2008+)
    and streams transaction logs, catalog data, and inventory state into Polars DataFrames.
    """

    def __init__(self, connection_string: Optional[str] = None):
        self.conn_str = connection_string or settings.mssql_connection_string
        self.engine = None
        if sa and settings.MSSQL_PASSWORD:
            try:
                # Format for SQLAlchemy with pyodbc
                host = f"{settings.MSSQL_SERVER}\\{settings.MSSQL_INSTANCE}" if settings.MSSQL_INSTANCE else settings.MSSQL_SERVER
                port = None if (settings.MSSQL_INSTANCE or "\\" in host) else settings.MSSQL_PORT
                query_params = {
                    "driver": settings.MSSQL_DRIVER,
                    "TrustServerCertificate": settings.MSSQL_TRUST_SERVER_CERTIFICATE,
                }
                if hasattr(settings, "MSSQL_ENCRYPT") and settings.MSSQL_ENCRYPT:
                    query_params["Encrypt"] = settings.MSSQL_ENCRYPT

                odbc_url = sa.engine.URL.create(
                    "mssql+pyodbc",
                    username=settings.MSSQL_USER,
                    password=settings.MSSQL_PASSWORD,
                    host=host,
                    port=port,
                    database=settings.MSSQL_DATABASE,
                    query=query_params,
                )
                self.engine = sa.create_engine(odbc_url)
                logger.info(f"Initialized MSSQL SQLAlchemy engine for {settings.MSSQL_DATABASE}")
            except Exception as e:
                logger.warning(f"Could not connect to MSSQL ({e}). Offline / synthetic mode available.")

    def get_active_2026_item_ids(self) -> set:
        """
        Retrieves the exact set of active item IDs where Del = 0 in BachirHome_2026.
        All other items across previous years are considered irrelevant.
        """
        db_2026 = "BachirHome_2026"
        conn_str = settings.get_mssql_connection_string(db_2026)
        query = "SELECT CAST(ItemID AS VARCHAR(50)) AS item_id FROM tblItem WITH (NOLOCK) WHERE Del = 0"
        try:
            if pyodbc:
                cn = pyodbc.connect(conn_str)
                cursor = cn.cursor()
                cursor.execute(query)
                ids = {str(r[0]) for r in cursor.fetchall()}
                cn.close()
                logger.info(f"Loaded {len(ids)} active items (Del=0) from {db_2026}")
                return ids
        except Exception as e:
            logger.error(f"Error querying active items from {db_2026}: {e}")
        return set()

    def fetch_transactions(
        self, 
        days_back: Optional[int] = None, 
        valid_item_ids: Optional[set] = None,
        use_history_cache: bool = True,
        refresh_history: bool = False
    ) -> pl.DataFrame:
        """
        Fetches historical transaction logs from all configured MSSQL databases (e.g. 2024, 2025, 2026),
        strictly filtered to only include items found in BachirHome_2026 with Del = 0.
        
        Optimization:
        Closed past fiscal years (2024, 2025) never change. If use_history_cache is True,
        they are loaded instantly from a local Parquet cache (data/historical_tx_2024_2025.parquet).
        Only the live current year (BachirHome_2026) is queried over the network each run.
        """
        from pathlib import Path
        import pandas as pd

        if valid_item_ids is None:
            valid_item_ids = self.get_active_2026_item_ids()

        databases = settings.database_list
        live_db = "BachirHome_2026"
        closed_dbs = [db for db in databases if db != live_db]
        
        cache_dir = Path("data")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "historical_tx_2024_2025.parquet"

        all_dfs = []

        # 1. Handle closed historical years (2024, 2025)
        if use_history_cache and not refresh_history and cache_file.exists() and not days_back:
            try:
                hist_df = pl.read_parquet(cache_file)
                # Verify and filter against valid_item_ids
                if valid_item_ids:
                    hist_df = hist_df.filter(pl.col("item_id").is_in(list(valid_item_ids)))
                all_dfs.append(hist_df)
                logger.info(f"Loaded {len(hist_df)} historical transactions for 2024-2025 from local cache ({cache_file}).")
            except Exception as e:
                logger.warning(f"Could not read historical cache: {e}. Will re-query MSSQL.")
                cache_file.unlink(missing_ok=True)

        # If historical years were not loaded from cache, query them from MSSQL
        if not all_dfs and closed_dbs and not days_back:
            closed_dfs = []
            for db in closed_dbs:
                conn_str = settings.get_mssql_connection_string(db)
                query = self._build_transaction_query(db, days_back=None)
                try:
                    if pyodbc:
                        cn = pyodbc.connect(conn_str)
                        pdf = pd.read_sql(query, cn)
                        cn.close()
                        if not pdf.empty:
                            if valid_item_ids:
                                pdf["item_id_str"] = pdf["item_id"].astype(str)
                                pdf = pdf[pdf["item_id_str"].isin(valid_item_ids)].drop(columns=["item_id_str"])
                            closed_dfs.append(pl.from_pandas(pdf))
                            logger.info(f"Loaded {len(pdf)} valid transactions for active 2026 items from {db}")
                except Exception as e:
                    logger.error(f"Error fetching transactions from closed DB {db}: {e}")
            
            if closed_dfs:
                merged_closed = pl.concat(closed_dfs)
                try:
                    merged_closed.write_parquet(cache_file)
                    logger.info(f"Saved {len(merged_closed)} historical rows to cache at {cache_file}.")
                except Exception as e:
                    logger.warning(f"Could not write cache file {cache_file}: {e}")
                all_dfs.append(merged_closed)

        # 2. Query the live database (BachirHome_2026) over the network
        dbs_to_query_live = [live_db] if live_db in databases else databases
        for db in dbs_to_query_live:
            conn_str = settings.get_mssql_connection_string(db)
            query = self._build_transaction_query(db, days_back=days_back)
            try:
                if pyodbc:
                    cn = pyodbc.connect(conn_str)
                    pdf = pd.read_sql(query, cn)
                    cn.close()
                    if not pdf.empty:
                        if valid_item_ids:
                            pdf["item_id_str"] = pdf["item_id"].astype(str)
                            pdf = pdf[pdf["item_id_str"].isin(valid_item_ids)].drop(columns=["item_id_str"])
                        all_dfs.append(pl.from_pandas(pdf))
                        logger.info(f"Loaded {len(pdf)} live transactions for active 2026 items from {db}")
            except Exception as e:
                logger.error(f"Error fetching live transactions from {db}: {e}")

        if all_dfs:
            combined = pl.concat(all_dfs).sort("timestamp")
            logger.info(f"Total merged transactions across all databases: {len(combined)}")
            return combined

        return self.generate_synthetic_transactions()

    def _build_transaction_query(self, db: str, days_back: Optional[int] = None) -> str:
        date_filter = ""
        if days_back:
            date_filter = f"""
            AND t.TrxDate >= DATEADD(day, -{days_back}, (
                SELECT ISNULL(MAX(TrxDate), GETDATE()) 
                FROM tblTransaction WITH (NOLOCK) 
                WHERE TrxTypeID = 2 AND Del = 0
            ))
            """

        return f"""
        SELECT 
            CAST(t.TrxID AS VARCHAR(50)) AS transaction_id,
            CAST(t.AcctID AS VARCHAR(50)) AS customer_id,
            CAST(s.ItemID AS VARCHAR(50)) AS item_id,
            t.TrxDate AS timestamp,
            CONVERT(VARCHAR(10), t.TrxDate, 120) AS date,
            CAST(d.Quantity AS FLOAT) AS quantity,
            CAST(ISNULL(d.UnitCost, 0.0) AS FLOAT) AS unit_cost,
            CAST(ISNULL(d.Price, 0.0) AS FLOAT) AS retail_price,
            CAST(ISNULL(d.Discount, 0.0) AS FLOAT) AS discount,
            '{db}' AS source_db
        FROM tblTransaction t WITH (NOLOCK)
        INNER JOIN tblTransactionDetail d WITH (NOLOCK) ON t.TrxID = d.TrxID
        INNER JOIN tblItemStock s WITH (NOLOCK) ON d.ItemStockID = s.ItemStockID
        INNER JOIN tblItem i WITH (NOLOCK) ON s.ItemID = i.ItemID
        INNER JOIN tblAccount a WITH (NOLOCK) ON t.AcctID = a.AcctID
        WHERE t.TrxTypeID = 2
          AND t.Del = 0
          AND d.Del = 0
          AND s.Del = 0
          AND i.Del = 0
          AND a.Del = 0
          AND a.AcctTypeID = 2
          {date_filter}
        ORDER BY t.TrxDate ASC
        """

    def fetch_catalog(self) -> pl.DataFrame:
        """
        Fetches item catalog, category taxonomy, prices, and stock strictly from BachirHome_2026 (Del=0):
        tblItem (Del=0) + tblItemClass (Del=0) + tblItemStock (Del=0).
        All other items not present in BachirHome_2026 are excluded.
        """
        target_db = "BachirHome_2026"
        conn_str = settings.get_mssql_connection_string(target_db)

        query = """
        SELECT 
            CAST(i.ItemID AS VARCHAR(50)) AS item_id,
            CAST(i.Code AS NVARCHAR(100)) AS code,
            CAST(i.ItemName AS NVARCHAR(255)) AS name,
            CAST(ISNULL(i.ClassID, 0) AS VARCHAR(50)) AS category_id,
            CAST(ISNULL(c.ItemClass, 'General') AS NVARCHAR(255)) AS category_name,
            CAST(ISNULL(stk.UnitCost, 0.0) AS FLOAT) AS unit_cost,
            CAST(ISNULL(stk.SRP, ISNULL(stk.Price, 0.0)) AS FLOAT) AS retail_price,
            CAST(
                CASE 
                    WHEN ISNULL(stk.SRP, ISNULL(stk.Price, 0.0)) > 0 
                    THEN (ISNULL(stk.SRP, ISNULL(stk.Price, 0.0)) - ISNULL(stk.UnitCost, 0.0)) / ISNULL(stk.SRP, ISNULL(stk.Price, 0.0))
                    ELSE 0.0 
                END AS FLOAT
            ) AS margin,
            CAST(ISNULL(stk.Qtty, 0) AS INT) AS current_stock,
            CAST(7 AS INT) AS supplier_lead_time_days,
            CAST(ISNULL(i.IsActive, 1) AS BIT) AS is_active
        FROM tblItem i WITH (NOLOCK)
        LEFT JOIN tblItemClass c WITH (NOLOCK) ON i.ClassID = c.ItemClassID AND c.Del = 0
        OUTER APPLY (
            SELECT TOP 1 UnitCost, SRP, Price, Qtty
            FROM tblItemStock s WITH (NOLOCK)
            WHERE s.ItemID = i.ItemID AND s.Del = 0
            ORDER BY s.ItemStockID DESC
        ) stk
        WHERE i.Del = 0
        ORDER BY i.ItemID ASC
        """

        try:
            import pandas as pd
            if pyodbc:
                cn = pyodbc.connect(conn_str)
                pdf = pd.read_sql(query, cn)
                cn.close()
                logger.info(f"Loaded master active catalog from {target_db} with Del=0: {len(pdf)} items")
                return pl.from_pandas(pdf)
        except Exception as e:
            logger.error(f"Error fetching catalog from {target_db}: {e}")

        return self.generate_synthetic_catalog()

    def fetch_accounts(self) -> pl.DataFrame:
        """
        Fetches active customer accounts across configured databases (Del=0 and AcctTypeID=2).
        Later databases take precedence for current balance and active state.
        """
        databases = settings.database_list
        acc_dict = {}

        query = """
        SELECT 
            CAST(AcctID AS VARCHAR(50)) AS customer_id,
            CAST(AcctName AS NVARCHAR(255)) AS customer_name,
            CAST(AcctNumber AS NVARCHAR(100)) AS account_number,
            CAST(AcctTypeID AS INT) AS account_type_id,
            CAST(ISNULL(Balance, 0.0) AS FLOAT) AS balance,
            CAST(ISNULL(IsActive, 1) AS BIT) AS is_active
        FROM tblAccount WITH (NOLOCK)
        WHERE Del = 0
          AND AcctTypeID = 2
        """

        for db in databases:
            conn_str = settings.get_mssql_connection_string(db)
            try:
                import pandas as pd
                if pyodbc:
                    cn = pyodbc.connect(conn_str)
                    pdf = pd.read_sql(query, cn)
                    cn.close()
                    for r in pdf.to_dict("records"):
                        acc_dict[str(r["customer_id"])] = r
            except Exception as e:
                logger.error(f"Error fetching accounts from {db}: {e}")

        if acc_dict:
            res = pl.DataFrame(list(acc_dict.values()))
            logger.info(f"Total merged customer accounts: {len(res)}")
            return res

        return pl.DataFrame()

    @staticmethod
    def generate_synthetic_catalog(n_items: int = 100) -> pl.DataFrame:
        """Generates realistic synthetic catalog data for prototyping."""
        np.random.seed(42)
        categories = ["Electronics", "Home & Kitchen", "Apparel", "Sports & Outdoors", "Beauty & Care"]
        
        items = []
        for i in range(1, n_items + 1):
            cat = np.random.choice(categories)
            cost = round(float(np.random.uniform(5.0, 150.0)), 2)
            margin_pct = np.random.uniform(0.2, 0.6)
            price = round(float(cost / (1.0 - margin_pct)), 2)
            lead_time = int(np.random.choice([3, 5, 7, 10, 14]))
            stock = int(np.random.poisson(lam=50))
            
            items.append({
                "item_id": f"ITEM_{i:04d}",
                "name": f"Product {cat} #{i}",
                "category_id": f"CAT_{categories.index(cat) + 1:02d}",
                "category_name": cat,
                "unit_cost": cost,
                "retail_price": price,
                "margin": round(float((price - cost) / price), 4),
                "current_stock": stock,
                "supplier_lead_time_days": lead_time,
                "is_active": True,
            })
        return pl.DataFrame(items)

    @staticmethod
    def generate_synthetic_transactions(
        n_transactions: int = 3000,
        n_customers: int = 200,
        n_items: int = 100,
        days_history: int = 180,
    ) -> pl.DataFrame:
        """Generates synthetic transaction logs with co-occurrence baskets & seasonality."""
        np.random.seed(42)
        catalog = MSSQLIngestionClient.generate_synthetic_catalog(n_items)
        item_price_map = {row["item_id"]: (row["unit_cost"], row["retail_price"]) for row in catalog.iter_rows(named=True)}
        
        records = []
        start_date = datetime.now() - timedelta(days=days_history)

        for tx_id in range(1, n_transactions + 1):
            cust_id = f"CUST_{np.random.randint(1, n_customers + 1):04d}"
            tx_time = start_date + timedelta(
                days=int(np.random.uniform(0, days_history)),
                seconds=int(np.random.uniform(0, 86400))
            )
            # Basket size (1 to 5 items)
            basket_size = int(np.random.choice([1, 2, 3, 4, 5], p=[0.4, 0.3, 0.15, 0.1, 0.05]))
            
            # Prefer items in similar clusters or categories
            base_item_idx = np.random.randint(1, n_items + 1)
            selected_items = [f"ITEM_{base_item_idx:04d}"]
            for _ in range(basket_size - 1):
                # Co-purchase bias towards adjacent items
                offset = int(np.random.choice([-2, -1, 1, 2, 5, 10]))
                co_item = max(1, min(n_items, base_item_idx + offset))
                selected_items.append(f"ITEM_{co_item:04d}")

            for item_id in set(selected_items):
                cost, price = item_price_map.get(item_id, (10.0, 20.0))
                # Optional price discount variation
                discount = round(float(np.random.choice([0.0, 0.05, 0.10, 0.20], p=[0.7, 0.15, 0.1, 0.05])), 2)
                effective_price = round(price * (1.0 - discount), 2)
                quantity = int(np.random.choice([1, 2, 3], p=[0.8, 0.15, 0.05]))

                records.append({
                    "transaction_id": f"TX_{tx_id:06d}",
                    "customer_id": cust_id,
                    "item_id": item_id,
                    "timestamp": tx_time,
                    "date": tx_time.strftime("%Y-%m-%d"),
                    "quantity": quantity,
                    "unit_cost": cost,
                    "retail_price": effective_price,
                    "discount": discount,
                })

        return pl.DataFrame(records)
