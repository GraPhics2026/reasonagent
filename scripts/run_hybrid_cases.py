"""Run the 4 hybrid test cases directly (no comparison with edit/gen).

Usage:
    python scripts/run_hybrid_cases.py --real-api       # real API calls
    python scripts/run_hybrid_cases.py --dry-run        # SVG placeholders
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from reason.hybrid_pipeline import run_hybrid_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all 4 hybrid test cases.")
    parser.add_argument("--real-api", dest="dry_run", action="store_false", default=True,
                        help="Use real MLLM + T2I API.")
    parser.add_argument("--cases", type=str, default=None,
                        help="Comma-separated case numbers, e.g. 1,2")
    parser.add_argument("--output", type=str, default="data/output/hybrid_cases",
                        help="Base output directory.")
    args = parser.parse_args()

    cases_path = _project_root / "data" / "input" / "hybrid" / "hybrid_cases.jsonl"
    if not cases_path.exists():
        print(f"Error: hybrid_cases.jsonl not found at {cases_path}")
        sys.exit(1)

    all_cases: list[dict] = []
    for line in cases_path.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            all_cases.append(json.loads(line))

    if args.cases:
        indices = {int(x.strip()) for x in args.cases.split(",")}
        selected = [(i, c) for i, c in enumerate(all_cases, start=1) if i in indices]
    else:
        selected = [(i, c) for i, c in enumerate(all_cases, start=1)]

    mode = "DRY-RUN" if args.dry_run else "REAL API"
    print(f"\n{'='*60}")
    print(f"Hybrid cases runner — {mode}")
    print(f"Cases: {len(selected)}")
    print(f"{'='*60}\n")

    results: list[dict] = []

    for case_idx, case in selected:
        image_rel = case["image"]
        instruction = case["instruction"]
        note = case.get("note", "")

        image_path = _project_root / image_rel
        if not image_path.exists():
            print(f"  [SKIP] Case {case_idx}: image not found at {image_path}")
            continue

        output_dir = Path(args.output) / f"case_{case_idx}"

        print(f"--- Case {case_idx}: {instruction} ---")
        print(f"       Note: {note}")
        print(f"       Image: {image_path}")
        print(f"       Output: {output_dir}")

        t0 = time.time()
        try:
            result = run_hybrid_pipeline(
                image_path=image_path,
                instruction=instruction,
                output_dir=output_dir,
                iterations=1,
                candidates=2,
                dry_run=args.dry_run,
                seed=42,
            )
            elapsed = round(time.time() - t0, 1)

            vqa_score = (
                result.vqa_result.get("score")
                if result.vqa_result
                else None
            )

            entry = {
                "case": case_idx,
                "instruction": instruction,
                "status": "success",
                "elapsed_seconds": elapsed,
                "final_image": result.final_image,
                "reasoning_type": result.reasoning_type,
                "vqa_score": vqa_score,
                "output_dir": str(output_dir),
            }
            results.append(entry)

            print(f"  [OK] Done in {elapsed}s | VQA={vqa_score} | type={result.reasoning_type}")
            print(f"       Final image: {result.final_image}")

        except Exception as exc:
            elapsed = round(time.time() - t0, 1)
            entry = {
                "case": case_idx,
                "instruction": instruction,
                "status": "failed",
                "elapsed_seconds": elapsed,
                "error": str(exc),
            }
            results.append(entry)
            print(f"  [FAIL] in {elapsed}s: {exc}")

        print()

    # Write summary
    summary_path = Path(args.output) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"{'='*60}")
    print(f"Summary written to {summary_path}")
    successes = sum(1 for r in results if r["status"] == "success")
    print(f"Results: {successes}/{len(results)} succeeded")
    for r in results:
        status_icon = "[OK]" if r["status"] == "success" else "[FAIL]"
        vqa = f" VQA={r.get('vqa_score')}" if r.get("vqa_score") is not None else ""
        print(f"  {status_icon} Case {r['case']}: {r['status']}{vqa} ({r.get('elapsed_seconds', '?')}s)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
