"""Batch run all hybrid test cases with real API (MLLM + T2I).

Usage:
    python batch_hybrid_cases.py [--iterations 1] [--candidates 2] [--seed 42]

Reads data/input/hybrid/hybrid_cases.jsonl and processes each case.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_hybrid")

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from reason.hybrid_pipeline import run_hybrid_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch run hybrid test cases.")
    parser.add_argument("--iterations", type=int, default=1, help="GenPilot optimization iterations.")
    parser.add_argument("--candidates", type=int, default=2, help="Candidates per GenPilot iteration.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for T2I.")
    parser.add_argument("--cases", type=str, default="data/input/hybrid/hybrid_cases.jsonl", help="Path to cases JSONL.")
    parser.add_argument("--output-base", type=str, default="data/output/hybrid", help="Base output directory.")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.exists():
        logger.error("Cases file not found: %s", cases_path)
        sys.exit(1)

    cases: list[dict[str, str]] = []
    with cases_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    logger.info("Loaded %d hybrid test cases from %s", len(cases), cases_path)
    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    summary_path = output_base / "_batch_summary.json"
    results: list[dict] = []

    for idx, case in enumerate(cases, start=1):
        image = case["image"]
        instruction = case["instruction"]
        note = case.get("note", "")
        output_dir = output_base / f"case_{idx}"

        logger.info("=" * 60)
        logger.info("Case %d/%d: %s", idx, len(cases), instruction)
        logger.info("  Image  : %s", image)
        logger.info("  Note   : %s", note)
        logger.info("  Output : %s", output_dir)

        image_path = Path(image)
        if not image_path.exists():
            logger.error("  SKIP — image not found: %s", image_path)
            results.append({
                "case": idx,
                "instruction": instruction,
                "status": "skipped",
                "error": f"Image not found: {image}",
            })
            continue

        t0 = time.perf_counter()
        try:
            result = run_hybrid_pipeline(
                image_path=image,
                instruction=instruction,
                output_dir=str(output_dir),
                iterations=args.iterations,
                candidates=args.candidates,
                dry_run=False,
                seed=args.seed,
            )
            elapsed = time.perf_counter() - t0
            result_dict = result.to_dict()
            logger.info("  ✅ DONE in %.1fs", elapsed)
            logger.info("  Final image : %s", result_dict.get("final_image", "N/A"))
            logger.info("  Scene prompt: %.120s...", result_dict.get("scene_prompt", "N/A"))
            if result_dict.get("vqa_result"):
                vqa = result_dict["vqa_result"]
                logger.info("  VQA score   : %s", vqa.get("score", "N/A"))
            results.append({
                "case": idx,
                "instruction": instruction,
                "note": note,
                "status": "success",
                "elapsed_seconds": round(elapsed, 1),
                "final_image": result_dict.get("final_image"),
                "scene_prompt": result_dict.get("scene_prompt"),
                "vqa_score": result_dict.get("vqa_result", {}).get("score") if result_dict.get("vqa_result") else None,
                "reasoning_type": result_dict.get("reasoning_type"),
                "output_dir": str(output_dir),
            })
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error("  ❌ FAILED after %.1fs: %s", elapsed, exc)
            import traceback
            traceback.print_exc()
            results.append({
                "case": idx,
                "instruction": instruction,
                "status": "failed",
                "error": str(exc),
                "elapsed_seconds": round(elapsed, 1),
            })

        logger.info("")

    # Write summary
    summary_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Print final summary
    logger.info("=" * 60)
    logger.info("BATCH SUMMARY")
    logger.info("=" * 60)
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    logger.info("Total: %d | ✅ Success: %d | ❌ Failed: %d | ⏭️ Skipped: %d",
                len(results), success, failed, skipped)
    logger.info("Summary written to: %s", summary_path)


if __name__ == "__main__":
    main()
