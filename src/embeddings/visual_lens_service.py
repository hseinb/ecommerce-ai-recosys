import base64
import concurrent.futures
import io
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
import numpy as np
import polars as pl
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field
from supabase import Client, create_client

from config.settings import settings

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


class LensVisionAnalysis(BaseModel):
    label: str = Field(description="Short name of the main retail product in the image")
    box_2d: List[int] = Field(
        default_factory=lambda: [0, 0, 1000, 1000],
        description="2D bounding box of the main product as [ymin, xmin, ymax, xmax] normalized to 0-1000",
    )
    keywords_en: List[str] = Field(
        default_factory=list,
        description="5 to 10 English keywords describing product type, category, material, shape, color, and room usage",
    )
    keywords_ar: List[str] = Field(
        default_factory=list,
        description="5 to 10 Arabic keywords (مرادفات وكلمات مفتاحية بالعربية) matching household/retail catalog names and descriptions",
    )
    visual_description_en: str = Field(
        default="",
        description="1-2 sentence English description of the product, material, and usage",
    )
    visual_description_ar: str = Field(
        default="",
        description="1-2 sentence Arabic description of the product, material, and usage",
    )


class VisualLensService:
    """
    Google Lens-like Visual & Keyword Search Engine for Recosys.
    - Indexes Cloudflare R2 product images (`public.item_images`) and product names/descriptions (`public.items`).
    - Syncs 768-D image & text embeddings into Supabase (`public.recosys_visual_index`).
    - Processes uploaded user images to return:
        1. Top 5-10 items by visual image embedding similarity
        2. Top 5-10 items by extracted image keywords vs item name & description similarity
    - Logs comparison results to `public.recosys_lens_searches`.
    """

    def __init__(self):
        self.project = settings.GCP_PROJECT or "gen-lang-client-0505999827"
        self.location = settings.GCP_LOCATION or "us-central1"
        self.use_vertex = settings.GEMINI_USE_VERTEX
        self.model_visual_emb = getattr(settings, "GEMINI_MODEL_VISUAL_EMBEDDING", "gemini-embedding-2-preview")
        self.model_reasoning = settings.GEMINI_MODEL_REASONING or "gemini-2.5-flash"
        self.embedding_dim = getattr(settings, "VISUAL_EMBEDDING_DIM", 768)

        self.cache_dir = Path("data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.img_cache_file = self.cache_dir / "gemini_visual_image_embeddings.parquet"
        self.txt_cache_file = self.cache_dir / "gemini_visual_text_embeddings.parquet"

        if genai is None:
            raise ImportError("google-genai SDK is required.")

        if self.use_vertex:
            self.client = genai.Client(
                vertexai=True,
                project=self.project,
                location=self.location,
            )
        else:
            api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
            self.client = genai.Client(api_key=api_key)

        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be configured in .env")
        self.supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

    # =========================================================================
    # 1. FETCH ITEMS & CLOUDFLARE R2 IMAGE LINKS FROM SUPABASE
    # =========================================================================
    def _fetch_all_rows(self, table: str, select_cols: str = "*", page_size: int = 1000) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        offset = 0
        while True:
            res = self.supabase.table(table).select(select_cols).range(offset, offset + page_size - 1).execute()
            batch = res.data or []
            rows.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size
        return rows

    @staticmethod
    def _extract_r2_urls(img_row: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """Extracts Cloudflare R2 thumb and view URLs from item_images JSONB slots."""
        for slot in ["hero1", "dim1", "angle1", "angle2", "angle3", "angle4", "angle5"]:
            val = img_row.get(slot)
            if isinstance(val, dict):
                thumb = val.get("thumb") or val.get("view")
                view = val.get("view") or val.get("thumb")
                if thumb or view:
                    return thumb, view
        return None, None

    def fetch_supabase_catalog_with_images(self) -> List[Dict[str, Any]]:
        """
        Fetches `public.items`, `public.item_class`, and `public.item_images` from Supabase
        and merges them with their Cloudflare R2 URLs and rich text descriptions.
        """
        logger.info("Fetching items, categories, and Cloudflare R2 image links from Supabase...")
        items = self._fetch_all_rows(
            "items",
            "item_id,code,item_name,class_id,description,material,house_rooms,tags,qtty,srp,is_active,image_updated_at",
        )
        classes = self._fetch_all_rows("item_class", "item_class_id,item_class,item_class_description")
        images = self._fetch_all_rows(
            "item_images",
            "item_id,hero1,dim1,angle1,angle2,angle3,angle4,angle5,updated_at",
        )

        class_map = {c["item_id"] if "item_id" in c else c["item_class_id"]: c.get("item_class", "General") for c in classes}
        image_map = {img["item_id"]: img for img in images}

        merged: List[Dict[str, Any]] = []
        for it in items:
            iid = int(it["item_id"])
            cat_name = class_map.get(it.get("class_id"), "General")
            img_row = image_map.get(iid, {})
            thumb_url, view_url = self._extract_r2_urls(img_row) if img_row else (None, None)

            item_name = (it.get("item_name") or "").strip()
            code = (it.get("code") or "").strip()
            description = (it.get("description") or "").strip()
            material = (it.get("material") or "").strip()
            house_rooms = (it.get("house_rooms") or "").strip()
            tags = (it.get("tags") or "").strip()

            parts = [f"Product: {item_name}"]
            if cat_name:
                parts.append(f"Category: {cat_name}")
            if description:
                parts.append(f"Description: {description}")
            if material:
                parts.append(f"Material: {material}")
            if house_rooms:
                parts.append(f"Rooms: {house_rooms}")
            if tags:
                parts.append(f"Tags: {tags}")
            if code:
                parts.append(f"Code: {code}")
            searchable_text = " | ".join(parts)

            merged.append({
                "item_id": iid,
                "code": code,
                "item_name": item_name,
                "category_name": cat_name,
                "description": description,
                "material": material,
                "house_rooms": house_rooms,
                "tags": tags,
                "r2_thumb_url": thumb_url,
                "r2_view_url": view_url,
                "searchable_text": searchable_text,
                "price": float(it.get("srp") or 0.0),
                "stock": float(it.get("qtty") or 0.0),
                "is_active": bool(it.get("is_active", True)),
                "image_source_updated_at": img_row.get("updated_at") or it.get("image_updated_at"),
            })

        with_r2 = sum(1 for m in merged if m["r2_thumb_url"])
        logger.info(f"Loaded {len(merged)} items from Supabase ({with_r2} with Cloudflare R2 images).")
        return merged

    # =========================================================================
    # 2. IMAGE CONVERSION & MULTIMODAL EMBEDDING HELPERS
    # =========================================================================
    @staticmethod
    def _to_jpeg_bytes(raw_bytes: bytes, max_side: int = 768) -> bytes:
        """Converts WebP/PNG/JPEG bytes to RGB JPEG bytes suitable for Gemini Embedding 2."""
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        return buf.getvalue()

    @staticmethod
    def _crop_bbox_jpeg(raw_bytes: bytes, box_2d: List[int], pad_ratio: float = 0.04) -> bytes:
        """Crops an image using normalized [ymin, xmin, ymax, xmax] coordinates (0-1000)."""
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        w, h = img.size
        if len(box_2d) == 4:
            ymin, xmin, ymax, xmax = box_2d
            # Ensure valid box
            if 0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000:
                box_h = ymax - ymin
                box_w = xmax - xmin
                # Only crop if box is not effectively the entire frame or tiny noise
                if (box_h * box_w) >= 2500 and (box_h < 980 or box_w < 980):
                    pad_y = int(box_h * pad_ratio)
                    pad_x = int(box_w * pad_ratio)
                    ymin = max(0, ymin - pad_y)
                    xmin = max(0, xmin - pad_x)
                    ymax = min(1000, ymax + pad_y)
                    xmax = min(1000, xmax + pad_x)
                    left = int((xmin / 1000.0) * w)
                    top = int((ymin / 1000.0) * h)
                    right = int((xmax / 1000.0) * w)
                    bottom = int((ymax / 1000.0) * h)
                    if right > left + 16 and bottom > top + 16:
                        img = img.crop((left, top, right, bottom))

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def embed_image_bytes(self, image_bytes: bytes) -> List[float]:
        """Generates a 768-D L2-normalized embedding for an image using gemini-embedding-2-preview."""
        jpg_bytes = self._to_jpeg_bytes(image_bytes)
        for attempt in range(5):
            try:
                res = self.client.models.embed_content(
                    model=self.model_visual_emb,
                    contents=[types.Part.from_bytes(data=jpg_bytes, mime_type="image/jpeg")],
                    config=types.EmbedContentConfig(output_dimensionality=self.embedding_dim),
                )
                vec = np.array(res.embeddings[0].values, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                return vec.tolist()
            except Exception as e:
                if attempt == 4:
                    raise e
                wait_s = 3.0 * (attempt + 1) if "429" in str(e) else 1.0 * (attempt + 1)
                time.sleep(wait_s)
        return []

    def embed_text(self, text: str, is_query: bool = False, title: Optional[str] = None) -> List[float]:
        """
        Generates a 768-D L2-normalized text embedding using gemini-embedding-2-preview.
        Uses asymmetric formatting for retrieval queries vs catalog documents.
        """
        if is_query:
            formatted = f"task: search result | query: {text}"
        else:
            doc_title = title if title else "none"
            formatted = f"title: {doc_title} | text: {text}"

        for attempt in range(5):
            try:
                res = self.client.models.embed_content(
                    model=self.model_visual_emb,
                    contents=[formatted],
                    config=types.EmbedContentConfig(output_dimensionality=self.embedding_dim),
                )
                vec = np.array(res.embeddings[0].values, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                return vec.tolist()
            except Exception as e:
                if attempt == 4:
                    raise e
                wait_s = 3.0 * (attempt + 1) if "429" in str(e) else 1.0 * (attempt + 1)
                time.sleep(wait_s)
        return []

    # =========================================================================
    # 3. CATALOG INDEXING PIPELINE (R2 IMAGES + ITEM DESCRIPTIONS -> SUPABASE)
    # =========================================================================
    def _load_cached_embeddings(self, path: Path) -> Dict[int, Tuple[List[float], str]]:
        """Loads {item_id: (embedding, url_or_text_hash)} from local Parquet cache."""
        cache: Dict[int, Tuple[List[float], str]] = {}
        if path.exists():
            try:
                df = pl.read_parquet(path)
                for r in df.iter_rows(named=True):
                    cache[int(r["item_id"])] = (list(r["embedding"]), str(r.get("source_key") or ""))
            except Exception as e:
                logger.warning(f"Could not read cache {path}: {e}")
        return cache

    def _save_cached_embeddings(self, path: Path, cache: Dict[int, Tuple[List[float], str]]) -> None:
        if not cache:
            return
        rows = [
            {"item_id": int(iid), "embedding": vec, "source_key": skey}
            for iid, (vec, skey) in cache.items()
        ]
        pl.DataFrame(rows).write_parquet(path)

    def build_and_sync_visual_index(
        self,
        limit: Optional[int] = None,
        only_with_images: bool = False,
        max_workers: int = 8,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Downloads R2 product thumbnails, generates 768-D image embeddings and 768-D text/description
        embeddings, caches locally in `data/`, and upserts to `public.recosys_visual_index` in Supabase.
        """
        catalog = self.fetch_supabase_catalog_with_images()
        if only_with_images:
            catalog = [c for c in catalog if c.get("r2_thumb_url")]
        if limit is not None:
            catalog = catalog[:limit]

        img_cache = {} if force_refresh else self._load_cached_embeddings(self.img_cache_file)
        txt_cache = {} if force_refresh else self._load_cached_embeddings(self.txt_cache_file)

        # 1. Identify items needing R2 image embeddings
        items_needing_img = [
            it for it in catalog
            if it.get("r2_thumb_url")
            and (it["item_id"] not in img_cache or img_cache[it["item_id"]][1] != it["r2_thumb_url"])
        ]
        # 2. Identify items needing text/description embeddings
        items_needing_txt = [
            it for it in catalog
            if it["item_id"] not in txt_cache or txt_cache[it["item_id"]][1] != it["searchable_text"]
        ]

        logger.info(
            f"Visual Index Plan: {len(catalog)} items targeted | "
            f"{len(items_needing_img)} R2 images to embed | "
            f"{len(items_needing_txt)} item descriptions to embed."
        )

        lock = threading.Lock()
        http_client = httpx.Client(timeout=20.0, follow_redirects=True)

        # --- Process Image Embeddings ---
        if items_needing_img:
            completed_img = 0

            def _embed_single_r2_item(it: Dict[str, Any]) -> None:
                nonlocal completed_img
                iid = it["item_id"]
                url = it["r2_thumb_url"] or it["r2_view_url"]
                try:
                    resp = http_client.get(url)
                    resp.raise_for_status()
                    vec = self.embed_image_bytes(resp.content)
                    if vec:
                        with lock:
                            img_cache[iid] = (vec, url)
                            completed_img += 1
                            if completed_img % 50 == 0 or completed_img == len(items_needing_img):
                                logger.info(f"Embedded {completed_img}/{len(items_needing_img)} R2 images...")
                                self._save_cached_embeddings(self.img_cache_file, img_cache)
                except Exception as e:
                    logger.warning(f"Skipping image embedding for item {iid} ({url}): {e}")

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                list(ex.map(_embed_single_r2_item, items_needing_img))
            self._save_cached_embeddings(self.img_cache_file, img_cache)

        # --- Process Text/Description Embeddings ---
        if items_needing_txt:
            completed_txt = 0

            def _embed_single_text_item(it: Dict[str, Any]) -> None:
                nonlocal completed_txt
                iid = it["item_id"]
                stext = it["searchable_text"]
                try:
                    vec = self.embed_text(stext, is_query=False, title=it["item_name"])
                    if vec:
                        with lock:
                            txt_cache[iid] = (vec, stext)
                            completed_txt += 1
                            if completed_txt % 100 == 0 or completed_txt == len(items_needing_txt):
                                logger.info(f"Embedded {completed_txt}/{len(items_needing_txt)} item descriptions...")
                                self._save_cached_embeddings(self.txt_cache_file, txt_cache)
                except Exception as e:
                    logger.warning(f"Failed text embedding for item {iid}: {e}")

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                list(ex.map(_embed_single_text_item, items_needing_txt))
            self._save_cached_embeddings(self.txt_cache_file, txt_cache)

        http_client.close()

        # --- Upsert into Supabase `public.recosys_visual_index` ---
        now_iso = datetime.now(timezone.utc).isoformat()
        upsert_rows: List[Dict[str, Any]] = []
        for it in catalog:
            iid = it["item_id"]
            img_vec = img_cache.get(iid, (None, ""))[0]
            txt_vec = txt_cache.get(iid, (None, ""))[0]
            if img_vec is None and txt_vec is None:
                continue
            upsert_rows.append({
                "item_id": iid,
                "code": it["code"],
                "item_name": it["item_name"],
                "category_name": it["category_name"],
                "description": it["description"],
                "material": it["material"],
                "house_rooms": it["house_rooms"],
                "tags": it["tags"],
                "r2_thumb_url": it["r2_thumb_url"],
                "r2_view_url": it["r2_view_url"],
                "searchable_text": it["searchable_text"],
                "image_embedding": img_vec,
                "text_embedding": txt_vec,
                "image_source_updated_at": it["image_source_updated_at"],
                "updated_at": now_iso,
            })

        batch_size = 150
        logger.info(f"Upserting {len(upsert_rows)} indexed items to Supabase `public.recosys_visual_index`...")
        for i in range(0, len(upsert_rows), batch_size):
            batch = upsert_rows[i : i + batch_size]
            self.supabase.table("recosys_visual_index").upsert(batch, on_conflict="item_id").execute()

        stats = {
            "total_catalog_processed": len(catalog),
            "upserted_to_supabase": len(upsert_rows),
            "with_image_embeddings": sum(1 for r in upsert_rows if r["image_embedding"] is not None),
            "with_text_embeddings": sum(1 for r in upsert_rows if r["text_embedding"] is not None),
        }
        logger.info(f"Visual Index Sync Complete: {stats}")
        return stats

    # =========================================================================
    # 4. VISION KEYWORD EXTRACTION & DUAL SIMILARITY SEARCH PIPELINE
    # =========================================================================
    def analyze_query_image(self, image_bytes: bytes) -> LensVisionAnalysis:
        """
        Uses Gemini Vision (`gemini-2.5-flash`) to:
        1. Detect the main product's 2D bounding box [ymin, xmin, ymax, xmax] (0-1000)
        2. Extract rich English AND Arabic keywords + descriptions for matching against `public.items`.
        """
        jpg_bytes = self._to_jpeg_bytes(image_bytes)
        prompt = (
            "You are a visual search assistant for a home, kitchenware, cleaning supplies, tools, and retail e-commerce catalog. "
            "Analyze the uploaded image and return strict JSON matching the schema:\n"
            "1. 'label': Concise name of the main retail item.\n"
            "2. 'box_2d': 2D bounding box of the primary product as [ymin, xmin, ymax, xmax] scaled 0 to 1000.\n"
            "3. 'keywords_en': 5-10 specific English keywords (product name, category, material, shape, function, room).\n"
            "4. 'keywords_ar': 5-10 specific Arabic keywords and common retail synonyms (بالعربية الفصحى والعامية السوقية مثل: قداحة، فوط تمسيح، مايكروفايبر، حبل غسيل، طقم، ستانلس ستيل، بلاستيك، مطبخ، تنظيف) so we can match Arabic product names and descriptions.\n"
            "5. 'visual_description_en': 1-2 sentence English description of the product.\n"
            "6. 'visual_description_ar': 1-2 sentence Arabic description of the product matching catalog style."
        )

        resp = self.client.models.generate_content(
            model=self.model_reasoning,
            contents=[
                types.Part.from_bytes(data=jpg_bytes, mime_type="image/jpeg"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=LensVisionAnalysis,
                temperature=0.1,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        try:
            return LensVisionAnalysis.model_validate_json(resp.text)
        except Exception:
            text = resp.text.strip()
            if "```" in text:
                text = text.split("```")[1].replace("json", "", 1).strip()
            return LensVisionAnalysis.model_validate(json.loads(text))

    @staticmethod
    def _load_image_input(image_input: Union[bytes, str, Path]) -> bytes:
        """Loads image bytes from raw bytes, local file path, URL, or base64 string."""
        if isinstance(image_input, bytes):
            return image_input
        if isinstance(image_input, Path):
            return image_input.read_bytes()
        if isinstance(image_input, str):
            s = image_input.strip()
            if s.startswith("http://") or s.startswith("https://"):
                resp = httpx.get(s, timeout=20.0, follow_redirects=True)
                resp.raise_for_status()
                return resp.content
            if os.path.exists(s):
                return Path(s).read_bytes()
            if s.startswith("data:image"):
                s = s.split(",", 1)[1]
            return base64.b64decode(s)
        raise ValueError(f"Unsupported image_input type: {type(image_input)}")

    def search_by_image(
        self,
        image_input: Union[bytes, str, Path],
        top_k: int = 10,
        active_only: bool = True,
        crop_detected_object: bool = True,
        save_search_to_supabase: bool = True,
    ) -> Dict[str, Any]:
        """
        End-to-end Google Lens pipeline when a user uploads an image:
        1. Extracts object bounding box + bilingual keywords (Arabic & English) via Gemini Vision.
        2. Crops the detected product and embeds the image (`gemini-embedding-2-preview`, 768-D).
        3. Embeds the extracted keywords & descriptions (`gemini-embedding-2-preview`, 768-D).
        4. Queries Supabase `match_items_by_lens` RPC to retrieve:
           - Top `top_k` (5-10) items by Image Embedding similarity
           - Top `top_k` (5-10) items by Text-to-Keywords similarity against item names & descriptions
        5. Saves the comparison result in `public.recosys_lens_searches`.
        """
        t0 = time.time()
        raw_bytes = self._load_image_input(image_input)

        # Step 1: Vision analysis (bounding box + Arabic/English keywords)
        logger.info("Analyzing uploaded image with Gemini Vision for bounding box & bilingual keywords...")
        analysis = self.analyze_query_image(raw_bytes)

        # Step 2: Crop image to detected product bounding box & compute 768-D Image Embedding
        if crop_detected_object and analysis.box_2d:
            query_img_bytes = self._crop_bbox_jpeg(raw_bytes, analysis.box_2d)
        else:
            query_img_bytes = self._to_jpeg_bytes(raw_bytes)

        all_keywords = analysis.keywords_ar + analysis.keywords_en
        keywords_query_str = " ".join(all_keywords)
        rich_keyword_query = (
            f"{analysis.label} | "
            f"Keywords AR: {', '.join(analysis.keywords_ar)} | "
            f"Keywords EN: {', '.join(analysis.keywords_en)} | "
            f"{analysis.visual_description_ar} | {analysis.visual_description_en}"
        )

        # Generate both embeddings concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            fut_img = ex.submit(self.embed_image_bytes, query_img_bytes)
            fut_txt = ex.submit(self.embed_text, rich_keyword_query, True)
            query_image_embedding = fut_img.result()
            query_text_embedding = fut_txt.result()

        # Step 3: Query Supabase RPC `match_items_by_lens`
        logger.info(f"Querying Supabase `match_items_by_lens` for top-{top_k} image & keyword matches...")
        rpc_res = self.supabase.rpc(
            "match_items_by_lens",
            {
                "query_image_embedding": query_image_embedding,
                "query_text_embedding": query_text_embedding,
                "query_keywords_text": keywords_query_str,
                "match_count": top_k,
                "active_only": active_only,
            },
        ).execute()

        payload = rpc_res.data or {}
        if isinstance(payload, list) and payload:
            payload = payload[0]

        image_matches = payload.get("image_matches", [])
        keyword_matches = payload.get("keyword_matches", [])

        elapsed_ms = round((time.time() - t0) * 1000, 1)

        # Step 4: Save comparison result in Supabase `public.recosys_lens_searches`
        search_id = None
        if save_search_to_supabase:
            try:
                insert_res = (
                    self.supabase.table("recosys_lens_searches")
                    .insert({
                        "query_label": analysis.label,
                        "extracted_keywords": all_keywords,
                        "visual_description_en": analysis.visual_description_en,
                        "visual_description_ar": analysis.visual_description_ar,
                        "detected_bbox": analysis.box_2d,
                        "top_image_matches": image_matches,
                        "top_keyword_matches": keyword_matches,
                    })
                    .execute()
                )
                if insert_res.data:
                    search_id = insert_res.data[0].get("id")
            except Exception as e:
                logger.warning(f"Could not log search to recosys_lens_searches: {e}")

        return {
            "search_id": search_id,
            "latency_ms": elapsed_ms,
            "query_analysis": {
                "label": analysis.label,
                "box_2d": analysis.box_2d,
                "keywords_en": analysis.keywords_en,
                "keywords_ar": analysis.keywords_ar,
                "visual_description_en": analysis.visual_description_en,
                "visual_description_ar": analysis.visual_description_ar,
            },
            "top_image_matches": image_matches,
            "top_keyword_matches": keyword_matches,
        }
