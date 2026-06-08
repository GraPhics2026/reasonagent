"""ReasonGenPilot unified entrypoint.

Dispatches to ``gen`` / ``edit`` / ``hybrid`` via :mod:`reason.router`.

A single ``--prompt`` is required for every mode:

- gen treats it as a descriptive text-to-image prompt
- edit / hybrid treat it as a counterfactual instruction

CLI examples
------------

::

    # auto mode (router decides):
    python pipeline.py --prompt "A windmill in a poppy field" --output data/output/demo

    # force a route:
    python pipeline.py --image data/input/edit/elephant_squirrel_grass.png \
        --prompt "如果大象和松鼠玩跷跷板会怎样" \
        --mode edit --output data/output/demo --real-api
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reason.edit_pipeline import run_edit_pipeline
from reason.gen_pipeline import run_gen_pipeline
from reason.hybrid_pipeline import run_hybrid_pipeline
from reason.router import route as router_route


def run_pipeline(
    prompt: str,
    image_path: str | Path | None = None,
    output_dir: str | Path = "data/output",
    mode: str = "auto",
    dry_run: bool | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Route the input through the appropriate pipeline and return its dict.

    ``mode`` defaults to ``"auto"`` (router decides). Force a route with
    ``"gen" | "edit" | "hybrid"``. ``prompt`` is required for every mode.
    """

    chosen = router_route(prompt=prompt, image_path=image_path, mode=mode)

    out = Path(output_dir)

    if chosen == "gen":
        result = run_gen_pipeline(
            prompt=prompt,
            output_dir=out,
            dry_run=dry_run,
            seed=seed,
        )
    elif chosen == "edit":
        result = run_edit_pipeline(
            image_path=image_path,
            instruction=prompt,
            output_dir=out,
            dry_run=dry_run,
            seed=seed,
        )
    elif chosen == "hybrid":
        result = run_hybrid_pipeline(
            image_path=image_path,
            instruction=prompt,
            output_dir=out,
            dry_run=dry_run,
            seed=seed,
        )
    else:  # pragma: no cover — router already validates
        raise ValueError(f"router: unknown route {chosen!r}.")

    return result.to_dict()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ReasonGenPilot unified pipeline (gen / edit / hybrid)."
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text input. Description for gen, counterfactual instruction for edit / hybrid.",
    )
    parser.add_argument(
        "--image",
        help="Reference image path. Required for edit / hybrid; ignored by gen.",
    )
    parser.add_argument("--output", default="data/output/demo", help="Output directory.")
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "gen", "edit", "hybrid"],
        help="Route selector. 'auto' lets reason.router decide.",
    )
    parser.add_argument(
        "--real-api",
        action="store_true",
        help="Call configured MLLM / T2I / Edit APIs instead of dry-run heuristics.",
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_pipeline(
        prompt=args.prompt,
        image_path=args.image,
        output_dir=args.output,
        mode=args.mode,
        dry_run=not args.real_api,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
