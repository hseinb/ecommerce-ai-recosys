"""
Visual Lens Pipeline CLI & Runner for Recosys.
1. Syncs Cloudflare R2 images (`public.item_images`) and product descriptions (`public.items`)
   into Supabase `public.recosys_visual_index`.
2. Accepts an image (local file path or URL) to run Dual Lens Comparison:
   - Top 5-10 items by Image Embedding similarity (`gemini-embedding-2-preview`)
   - Top 5-10 items by Extracted Keywords vs Item Names & Descriptions similarity
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from loguru import logger
from src.embeddings.visual_lens_service import VisualLensService


def print_lens_results(res: dict) -> None:
    qa = res.get("query_analysis", {})
    print("\n" + "=" * 85)
    print(f"🔍 LENS VISUAL ANALYSIS (Search ID: {res.get('search_id')} | {res.get('latency_ms')} ms)")
    print("=" * 85)
    print(f"Detected Product : {qa.get('label')}")
    print(f"Bounding Box     : {qa.get('box_2d')}")
    print(f"Keywords (EN)    : {', '.join(qa.get('keywords_en', []))}")
    print(f"Keywords (AR)    : {', '.join(qa.get('keywords_ar', []))}")
    print(f"Description (EN) : {qa.get('visual_description_en')}")
    print(f"Description (AR) : {qa.get('visual_description_ar')}")

    img_matches = res.get("top_image_matches", [])
    print("\n" + "-" * 85)
    print(f"🖼️  TOP {len(img_matches)} MATCHES BY IMAGE EMBEDDING SIMILARITY (Cloudflare R2)")
    print("-" * 85)
    for m in img_matches:
        print(
            f"  #{m.get('rank'):<2} | Score: {float(m.get('similarity_score', 0)):.4f} "
            f"| ID: {m.get('item_id'):<6} | Code: {str(m.get('code') or ''):<10} "
            f"| Price: ${float(m.get('price') or 0):.2f} | Stock: {float(m.get('stock') or 0):.0f} "
            f"| {m.get('item_name')} ({m.get('category_name')})"
        )

    kw_matches = res.get("top_keyword_matches", [])
    print("\n" + "-" * 85)
    print(f"🏷️  TOP {len(kw_matches)} MATCHES BY KEYWORDS-TO-DESCRIPTION SIMILARITY")
    print("-" * 85)
    for m in kw_matches:
        print(
            f"  #{m.get('rank'):<2} | Score: {float(m.get('similarity_score', 0)):.4f} "
            f"(Sem: {float(m.get('semantic_score', 0)):.4f}, Lex: {float(m.get('lexical_score', 0)):.4f}) "
            f"| ID: {m.get('item_id'):<6} | Code: {str(m.get('code') or ''):<10} "
            f"| {m.get('item_name')} ({m.get('category_name')})"
        )
    print("=" * 85 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Recosys Google Lens Visual & Keyword Search Pipeline")
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Index Cloudflare R2 item images and descriptions into Supabase recosys_visual_index",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of catalog items to index during --sync (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent workers for R2 image downloading and Gemini embedding",
    )
    parser.add_argument(
        "--only-with-images",
        action="store_true",
        help="Only index items that have a Cloudflare R2 image link",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path or URL to a query image to compare against the database",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top matches to return per modality (default: 10)",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to save the JSON search response",
    )

    args = parser.parse_args()
    service = VisualLensService()

    if args.sync:
        stats = service.build_and_sync_visual_index(
            limit=args.limit,
            only_with_images=args.only_with_images,
            max_workers=args.workers,
        )
        logger.info(f"Sync summary: {stats}")

    if args.image:
        result = service.search_by_image(
            image_input=args.image,
            top_k=args.top_k,
        )
        print_lens_results(result)
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"Saved search output to {args.json_out}")


if __name__ == "__main__":
    main()
