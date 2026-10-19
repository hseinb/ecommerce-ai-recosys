import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import polars as pl
from loguru import logger

import concurrent.futures
import threading
from config.settings import settings

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


class GeminiService:
    """
    Direct Google Cloud / Vertex AI integration using postpay billing project:
    Project: gen-lang-client-0505999827
    Models:
      - Embeddings: text-embedding-004
      - Reasoning & Pitches: gemini-2.5-flash

    STRICT MODE: No synthetic fallbacks. If any Gemini API call fails, it raises an exception immediately.
    """

    def __init__(self):
        self.project = settings.GCP_PROJECT or "gen-lang-client-0505999827"
        self.location = settings.GCP_LOCATION or "us-central1"
        self.use_vertex = settings.GEMINI_USE_VERTEX
        self.model_embedding = settings.GEMINI_MODEL_EMBEDDING or "text-embedding-004"
        self.model_reasoning = settings.GEMINI_MODEL_REASONING or "gemini-2.5-flash"
        
        self.cache_dir = Path("data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if genai is None:
            raise ImportError("google-genai SDK is not installed in the active environment.")

        logger.info(f"Connecting to Vertex AI (Project: {self.project}, Location: {self.location})...")
        try:
            if self.use_vertex:
                self.client = genai.Client(
                    vertexai=True,
                    project=self.project,
                    location=self.location
                )
            else:
                api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
                self.client = genai.Client(api_key=api_key)

            logger.info(f"Connected to Gemini on Google Cloud project {self.project} successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI client on project {self.project}: {e}")
            raise e

    # =========================================================================
    # 1. ITEM EMBEDDINGS (text-embedding-004)
    # =========================================================================
    def get_item_embeddings(self, catalog_df: pl.DataFrame, force_refresh: bool = False) -> Dict[str, np.ndarray]:
        """
        Generates dense 768-D semantic embeddings via text-embedding-004 on Vertex AI.
        Caches results to data/gemini_item_embeddings.parquet.
        """
        cache_file = self.cache_dir / "gemini_item_embeddings.parquet"
        cached_dict: Dict[str, np.ndarray] = {}

        if not force_refresh and cache_file.exists():
            try:
                cached_df = pl.read_parquet(cache_file)
                for row in cached_df.iter_rows(named=True):
                    cached_dict[str(row["item_id"])] = np.array(row["embedding"], dtype=np.float32)
                logger.info(f"Loaded {len(cached_dict)} Gemini embeddings from local cache.")
            except Exception as e:
                logger.warning(f"Error reading embedding cache: {e}. Will re-generate.")

        items_to_embed = []
        for row in catalog_df.iter_rows(named=True):
            iid = str(row["item_id"])
            if iid not in cached_dict:
                items_to_embed.append(row)

        if items_to_embed:
            logger.info(f"Generating live Gemini embeddings for {len(items_to_embed)} items via {self.model_embedding}...")
            new_embeddings = self._generate_embeddings_batch(items_to_embed)
            cached_dict.update(new_embeddings)

            # Persist to cache
            save_data = [
                {"item_id": k, "embedding": v.tolist()}
                for k, v in cached_dict.items()
            ]
            pl.DataFrame(save_data).write_parquet(cache_file)
            logger.info(f"Saved {len(cached_dict)} verified Gemini embeddings to {cache_file}.")

        return cached_dict

    def _generate_embeddings_batch(self, items: List[Dict[str, Any]], batch_size: int = 100) -> Dict[str, np.ndarray]:
        """
        Batches items into text-embedding-004 API requests.
        Strict: Raises exception if Gemini fails.
        """
        embeddings_map = {}
        total = len(items)

        for i in range(0, total, batch_size):
            batch = items[i:i + batch_size]
            texts = [
                f"Product: {it.get('name', '')}. Category: {it.get('category_name', '')}. Code: {it.get('code', '')}."
                for it in batch
            ]
            
            logger.info(f"Embedding batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size} ({len(batch)} items)...")
            try:
                resp = self.client.models.embed_content(
                    model=self.model_embedding,
                    contents=texts,
                )
                
                # In Vertex AI, resp.embeddings is a list of ContentEmbedding objects
                if hasattr(resp, "embeddings") and resp.embeddings:
                    for idx, emb_obj in enumerate(resp.embeddings):
                        iid = str(batch[idx]["item_id"])
                        embeddings_map[iid] = np.array(emb_obj.values, dtype=np.float32)
                elif hasattr(resp, "embedding") and resp.embedding:
                    iid = str(batch[0]["item_id"])
                    embeddings_map[iid] = np.array(resp.embedding.values, dtype=np.float32)
                else:
                    raise RuntimeError(f"Unexpected response format from {self.model_embedding}: {resp}")

            except Exception as e:
                logger.error(f"Gemini embedding API call failed on batch {i//batch_size + 1}: {e}")
                raise e  # Fail loudly - NO fallback!

            time.sleep(0.2)  # Avoid rate spikes

        return embeddings_map

    # =========================================================================
    # 2. ITEM SALES PITCHES & MERCHANDISING NOTES (gemini-2.5-flash)
    # =========================================================================
    def get_item_sales_pitches(self, catalog_df: pl.DataFrame, force_refresh: bool = False) -> Dict[str, Dict[str, str]]:
        """
        Uses gemini-2.5-flash to generate B2B sales pitches and merchandising notes for each product.
        Strict: Raises exception on failure.
        """
        cache_file = self.cache_dir / "gemini_item_pitches.parquet"
        cached_dict: Dict[str, Dict[str, str]] = {}

        if not force_refresh and cache_file.exists():
            try:
                cached_df = pl.read_parquet(cache_file)
                for row in cached_df.iter_rows(named=True):
                    cached_dict[str(row["item_id"])] = {
                        "sales_pitch": row.get("sales_pitch", ""),
                        "merchandising_notes": row.get("merchandising_notes", "")
                    }
                logger.info(f"Loaded {len(cached_dict)} sales pitches from local cache.")
            except Exception as e:
                logger.warning(f"Error reading pitches cache: {e}. Will re-generate.")

        items_missing = []
        for row in catalog_df.iter_rows(named=True):
            iid = str(row["item_id"])
            if iid not in cached_dict:
                items_missing.append(row)

        if items_missing:
            logger.info(f"Generating live Gemini sales pitches for {len(items_missing)} items via {self.model_reasoning}...")
            new_pitches = self._generate_pitches_batch(items_missing, cached_dict=cached_dict)
            cached_dict.update(new_pitches)

            save_data = [
                {
                    "item_id": k,
                    "sales_pitch": v["sales_pitch"],
                    "merchandising_notes": v["merchandising_notes"]
                }
                for k, v in cached_dict.items()
            ]
            pl.DataFrame(save_data).write_parquet(cache_file)
            logger.info(f"Saved {len(cached_dict)} verified Gemini pitches to {cache_file}.")

        return cached_dict

    def _generate_pitches_batch(
        self,
        items: List[Dict[str, Any]],
        cached_dict: Dict[str, Dict[str, str]],
        batch_size: int = 40,
        max_workers: int = 6
    ) -> Dict[str, Dict[str, str]]:
        pitches_map = {}
        total = len(items)
        batches = [items[i:i + batch_size] for i in range(0, total, batch_size)]
        num_batches = len(batches)
        logger.info(f"Generating pitches across {num_batches} batches using {max_workers} concurrent workers...")

        lock = threading.Lock()
        cache_file = self.cache_dir / "gemini_item_pitches.parquet"

        def _process_single_batch(batch_idx: int, batch: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
            batch_input = [
                {
                    "id": str(it["item_id"]),
                    "name": str(it.get("name") or "").replace('"', " in."),
                    "category": it.get("category_name"),
                    "price": float(it.get("retail_price") or 0.0),
                    "stock": int(it.get("current_stock") or 0),
                    "margin": round(float(it.get("margin") or 0.0), 2)
                }
                for it in batch
            ]

            prompt = (
                "You are an expert luxury tableware and home retail merchandising director. "
                "For each of the items provided, generate:\n"
                "1. 'sales_pitch': 1 concise, compelling sentence for B2B sales reps to pitch this product for re-orders or upselling.\n"
                "2. 'merchandising_notes': 1 short sentence on tabletop pairing or catalog role.\n\n"
                "CRITICAL: Do NOT use unescaped double quotes inside text values. Use single quotes if needed. "
                "Return a strict JSON array of objects with keys: 'id', 'sales_pitch', 'merchandising_notes'.\n\n"
                f"Items:\n{json.dumps(batch_input, ensure_ascii=False)}"
            )

            last_err = None
            for attempt in range(3):
                try:
                    logger.info(f"Submitting pitches batch {batch_idx + 1}/{num_batches} (attempt {attempt + 1})...")
                    resp = self.client.models.generate_content(
                        model=self.model_reasoning,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json"
                        ) if types else None
                    )

                    text = resp.text.strip()
                    if "```json" in text:
                        text = text.split("```json")[1].split("```")[0].strip()
                    elif "```" in text:
                        text = text.split("```")[1].split("```")[0].strip()

                    try:
                        parsed = json.loads(text)
                    except json.JSONDecodeError:
                        import re
                        cleaned = re.sub(r',\s*([}\]])', r'\1', text)
                        parsed = json.loads(cleaned)

                    batch_res = {}
                    for entry in parsed:
                        eid = str(entry.get("id"))
                        batch_res[eid] = {
                            "sales_pitch": entry.get("sales_pitch", ""),
                            "merchandising_notes": entry.get("merchandising_notes", "")
                        }

                    for it in batch:
                        iid = str(it["item_id"])
                        if iid not in batch_res:
                            batch_res[iid] = {
                                "sales_pitch": f"Premium {it.get('category_name', 'tableware')} collection piece.",
                                "merchandising_notes": f"Pairs with coordinating {it.get('category_name', 'dining')} sets."
                            }

                    with lock:
                        cached_dict.update(batch_res)
                        try:
                            save_rows = [
                                {"item_id": k, "sales_pitch": v["sales_pitch"], "merchandising_notes": v["merchandising_notes"]}
                                for k, v in cached_dict.items()
                            ]
                            pl.DataFrame(save_rows).write_parquet(cache_file)
                        except Exception as serr:
                            logger.warning(f"Error saving incremental parquet: {serr}")

                    logger.info(f"Completed and persisted pitches batch {batch_idx + 1}/{num_batches}.")
                    return batch_res

                except Exception as e:
                    last_err = e
                    logger.warning(f"Batch {batch_idx + 1} attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(1.0)

            logger.error(f"Gemini pitch batch {batch_idx + 1} failed after 3 attempts: {last_err}")
            raise last_err  # Fail loudly - NO fallback!

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_process_single_batch, idx, b): idx 
                for idx, b in enumerate(batches)
            }
            batch_errors = []
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    res = future.result()
                    pitches_map.update(res)
                except Exception as ex:
                    batch_errors.append((idx, ex))

            if batch_errors:
                first_idx, first_ex = batch_errors[0]
                logger.error(f"{len(batch_errors)} pitch batches failed! First error on batch {first_idx + 1}: {first_ex}")
                raise first_ex  # Fail loudly - NO fallback!

        return pitches_map

    # =========================================================================
    # 3. CLUSTER MERCHANDISING NAMES (gemini-2.5-flash)
    # =========================================================================
    def generate_cluster_names(self, cluster_items_map: Dict[int, List[Dict[str, Any]]]) -> Dict[int, str]:
        """
        Uses gemini-2.5-flash to name each of the 8 clusters based on member items.
        Strict: Raises exception on failure.
        """
        cluster_summaries = {}
        for cid, items in cluster_items_map.items():
            sample_names = [str(it.get("name", "")).replace('"', " ") for it in items[:8] if it.get("name")]
            cluster_summaries[f"Cluster_{cid}"] = sample_names

        prompt = (
            "You are a luxury retail merchandising executive. Given these 8 clusters of home & tableware products, "
            "provide a concise, elegant 2-4 word merchandising collection name for each (e.g. 'Fine Porcelain & Banquet Dining', "
            "'Commercial Culinary Cookware', 'Artisan Crystal & Barware').\n\n"
            "Return ONLY a valid JSON object mapping 'Cluster_0', 'Cluster_1', etc. to the collection name string.\n\n"
            f"Clusters:\n{json.dumps(cluster_summaries, ensure_ascii=False)}"
        )

        logger.info(f"Generating cluster merchandising names via {self.model_reasoning}...")
        try:
            resp = self.client.models.generate_content(
                model=self.model_reasoning,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ) if types else None
            )

            text = resp.text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                import re
                cleaned = re.sub(r',\s*([}\]])', r'\1', text)
                data = json.loads(cleaned)

            cluster_names = {}
            for k, v in data.items():
                cid = int(k.replace("Cluster_", "").strip())
                cluster_names[cid] = str(v)

            logger.info(f"Gemini successfully named {len(cluster_names)} merchandising clusters: {cluster_names}")
            return cluster_names
        except Exception as e:
            logger.error(f"Gemini cluster naming failed: {e}")
            raise e  # Fail loudly - NO fallback!

    # =========================================================================
    # 4. CUSTOMER BUYER PERSONAS & SALES NOTES (gemini-2.5-flash)
    # =========================================================================
    def get_customer_personas(
        self,
        accounts_df: pl.DataFrame,
        transactions_df: pl.DataFrame,
        force_refresh: bool = False
    ) -> Dict[str, Dict[str, str]]:
        """
        Uses gemini-2.5-flash to generate buyer personas and sales guidance for all customer accounts.
        Strict: Raises exception on failure.
        """
        cache_file = self.cache_dir / "gemini_customer_personas.parquet"
        cached_dict: Dict[str, Dict[str, str]] = {}

        if not force_refresh and cache_file.exists():
            try:
                cached_df = pl.read_parquet(cache_file)
                for row in cached_df.iter_rows(named=True):
                    cached_dict[str(row["customer_id"])] = {
                        "buyer_persona": row.get("buyer_persona", ""),
                        "sales_notes": row.get("sales_notes", "")
                    }
                logger.info(f"Loaded {len(cached_dict)} customer personas from local cache.")
            except Exception as e:
                logger.warning(f"Error reading persona cache: {e}. Will re-generate.")

        missing_customers = []
        for row in accounts_df.iter_rows(named=True):
            cid = str(row["customer_id"])
            if cid not in cached_dict:
                missing_customers.append(row)

        if missing_customers:
            logger.info(f"Generating live Gemini personas for {len(missing_customers)} customers via {self.model_reasoning}...")
            
            # Aggregate stats per customer
            tx_summary = {}
            for row in transactions_df.iter_rows(named=True):
                c_id = str(row["customer_id"])
                if c_id not in tx_summary:
                    tx_summary[c_id] = {"count": 0, "total_qty": 0.0, "total_val": 0.0, "sample_items": []}
                tx_summary[c_id]["count"] += 1
                qty = float(row.get("quantity") or 0.0)
                price = float(row.get("retail_price") or 0.0)
                tx_summary[c_id]["total_qty"] += qty
                tx_summary[c_id]["total_val"] += qty * price

            # Batch customers in groups of 30 for LLM profiling
            batch_size = 30
            batches = [missing_customers[i:i + batch_size] for i in range(0, len(missing_customers), batch_size)]
            num_batches = len(batches)
            logger.info(f"Generating personas across {num_batches} batches with 4 concurrent workers...")

            persona_lock = threading.Lock()

            def _process_persona_batch(batch_idx: int, batch: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
                batch_data = []
                for c in batch:
                    cid = str(c["customer_id"])
                    stats = tx_summary.get(cid, {"count": 0, "total_qty": 0.0, "total_val": 0.0})
                    cname = str(c.get("customer_name") or f"Account #{cid}").replace('"', " ")
                    batch_data.append({
                        "id": cid,
                        "name": cname,
                        "account_number": str(c.get("account_number") or ""),
                        "orders_count": stats["count"],
                        "total_volume_units": round(stats["total_qty"], 1),
                        "total_spend_usd": round(stats["total_val"], 2),
                    })

                prompt = (
                    "You are a B2B sales director in the luxury hospitality & tableware industry. "
                    "For each customer account below, provide:\n"
                    "1. 'buyer_persona': 1 sentence characterizing their purchasing scale and business profile (e.g. 'High-volume wholesale banquet buyer', 'Specialty boutique retail partner').\n"
                    "2. 'sales_notes': 1 sentence of practical action guidance for sales reps on re-ordering or cross-selling.\n\n"
                    "CRITICAL: Do NOT use unescaped double quotes inside text values. Use single quotes if needed. "
                    "Return a strict JSON array of objects with keys: 'id', 'buyer_persona', 'sales_notes'.\n\n"
                    f"Customers:\n{json.dumps(batch_data, ensure_ascii=False)}"
                )

                last_err = None
                for attempt in range(3):
                    try:
                        logger.info(f"Submitting personas batch {batch_idx + 1}/{num_batches} (attempt {attempt + 1})...")
                        resp = self.client.models.generate_content(
                            model=self.model_reasoning,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json"
                            ) if types else None
                        )

                        text = resp.text.strip()
                        if "```json" in text:
                            text = text.split("```json")[1].split("```")[0].strip()
                        elif "```" in text:
                            text = text.split("```")[1].split("```")[0].strip()

                        try:
                            parsed = json.loads(text)
                        except json.JSONDecodeError:
                            import re
                            cleaned = re.sub(r',\s*([}\]])', r'\1', text)
                            parsed = json.loads(cleaned)

                        batch_res = {}
                        for entry in parsed:
                            cid = str(entry.get("id"))
                            batch_res[cid] = {
                                "buyer_persona": entry.get("buyer_persona", ""),
                                "sales_notes": entry.get("sales_notes", "")
                            }

                        for c in batch:
                            cid = str(c["customer_id"])
                            if cid not in batch_res:
                                batch_res[cid] = {
                                    "buyer_persona": "Commercial wholesale tableware account with recurring replenishment orders.",
                                    "sales_notes": "Present seasonal banquet bundles and monitor reorder point thresholds."
                                }

                        with persona_lock:
                            cached_dict.update(batch_res)
                            try:
                                save_rows = [
                                    {"customer_id": k, "buyer_persona": v["buyer_persona"], "sales_notes": v["sales_notes"]}
                                    for k, v in cached_dict.items()
                                ]
                                pl.DataFrame(save_rows).write_parquet(cache_file)
                            except Exception as serr:
                                logger.warning(f"Error saving persona parquet: {serr}")

                        logger.info(f"Completed and persisted personas batch {batch_idx + 1}/{num_batches}.")
                        return batch_res

                    except Exception as e:
                        last_err = e
                        logger.warning(f"Persona batch {batch_idx + 1} attempt {attempt + 1} failed: {e}. Retrying...")
                        time.sleep(1.0)

                logger.error(f"Gemini persona batch {batch_idx + 1} failed after 3 attempts: {last_err}")
                raise last_err  # Fail loudly - NO fallback!

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_idx = {
                    executor.submit(_process_persona_batch, idx, b): idx
                    for idx, b in enumerate(batches)
                }
                batch_errors = []
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        res = future.result()
                        cached_dict.update(res)
                    except Exception as ex:
                        batch_errors.append((idx, ex))

                if batch_errors:
                    first_idx, first_ex = batch_errors[0]
                    logger.error(f"{len(batch_errors)} persona batches failed! First error on batch {first_idx + 1}: {first_ex}")
                    raise first_ex  # Fail loudly - NO fallback!

            save_data = [
                {
                    "customer_id": k,
                    "buyer_persona": v["buyer_persona"],
                    "sales_notes": v["sales_notes"]
                }
                for k, v in cached_dict.items()
            ]
            pl.DataFrame(save_data).write_parquet(cache_file)
            logger.info(f"Saved {len(cached_dict)} verified Gemini customer personas to {cache_file}.")

        return cached_dict
