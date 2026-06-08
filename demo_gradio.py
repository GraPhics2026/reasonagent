"""Gradio demo for ReasonGenPilot — gen / edit / hybrid in one UI.

Run with::

    pip install gradio
    python demo_gradio.py

A single ``Prompt`` field is required for every mode:

- gen treats it as a descriptive text-to-image prompt
- edit / hybrid treat it as a counterfactual instruction

Optional ``Reference image`` is required by edit / hybrid; ignored by gen.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import gradio as gr  # type: ignore[import-not-found]
from PIL import Image

from reason.edit_pipeline import run_edit_pipeline
from reason.gen_pipeline import run_gen_pipeline
from reason.hybrid_pipeline import run_hybrid_pipeline
from reason.router import route as router_route


DEMO_OUTPUT_ROOT = Path("data/output/demo")
DEMO_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def _save_uploaded_image(image: Image.Image | None, output_dir: Path) -> Path | None:
    if image is None:
        return None
    target = output_dir / "input.png"
    image.save(str(target))
    return target


def _format_reasoning(chain: list[str] | None) -> str:
    if not chain:
        return "(no reasoning chain)"
    return "\n".join(f"{i + 1}. {step}" for i, step in enumerate(chain))


def _format_vqa(vqa: dict[str, Any] | None) -> str:
    if not vqa:
        return "(no VQA result)"
    score = vqa.get("score")
    passed = vqa.get("passed", []) or []
    errors = vqa.get("errors", []) or []
    summary = vqa.get("summary", "")
    parts = [f"Score: {score}"]
    if summary:
        parts.append(f"Summary: {summary}")
    if passed:
        parts.append("Passed:\n" + "\n".join(f"  - {p}" for p in passed))
    if errors:
        parts.append("Errors:\n" + "\n".join(f"  - {e}" for e in errors))
    return "\n".join(parts)


def dispatch(
    mode: str,
    prompt: str,
    image: Image.Image | None,
    use_real_api: bool,
) -> tuple[Any, Any, str, str, str]:
    """Run the chosen route and return UI-friendly fields.

    Returns ``(before_image, after_image, route_label, reasoning_text, result_json)``.
    """

    output_dir = DEMO_OUTPUT_ROOT / f"run_{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_image = _save_uploaded_image(image, output_dir) if image is not None else None

    try:
        chosen = router_route(prompt=prompt, image_path=saved_image, mode=mode)
    except (ValueError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc

    dry_run = not use_real_api

    if chosen == "gen":
        r = run_gen_pipeline(prompt=prompt, output_dir=output_dir, dry_run=dry_run)
        before = None
    elif chosen == "edit":
        r = run_edit_pipeline(
            image_path=saved_image,
            instruction=prompt,
            output_dir=output_dir,
            min_iterations=1,  # demo: don't force the second loop
            dry_run=dry_run,
        )
        before = str(Path(r.image_before).resolve())
    elif chosen == "hybrid":
        r = run_hybrid_pipeline(
            image_path=saved_image,
            instruction=prompt,
            output_dir=output_dir,
            dry_run=dry_run,
        )
        before = str(Path(r.image_before).resolve())
    else:  # pragma: no cover
        raise gr.Error(f"router: unknown route {chosen!r}.")

    after = str(Path(r.final_image).resolve())
    payload = r.to_dict()
    reasoning_text = _format_reasoning(payload.get("reasoning_chain"))
    vqa_text = _format_vqa(payload.get("vqa_result"))
    full_text = (
        f"=== Reasoning ===\n{reasoning_text}\n\n"
        f"=== VQA ===\n{vqa_text}\n\n"
        f"=== Output dir ===\n{output_dir}"
    )
    result_json = json.dumps(payload, ensure_ascii=False, indent=2)
    return before, after, f"route = {chosen}", full_text, result_json


def _examples() -> list[list[Any]]:
    rows: list[list[Any]] = []
    rows.append([
        "gen",
        "A grass field filled with red poppies and yellow daisies beside a wooden windmill.",
        None,
        False,
    ])
    edit_img = Path("data/input/edit/elephant_squirrel_grass.png")
    if edit_img.exists():
        rows.append([
            "edit",
            "大象和松鼠玩跷跷板会怎样呢?",
            str(edit_img),
            False,
        ])
    hybrid_img = Path("data/input/hybrid/bouquet.png")
    if hybrid_img.exists():
        rows.append([
            "hybrid",
            "如果这一束花分成两束独立的玫瑰，会是什么样子？",
            str(hybrid_img),
            False,
        ])
    return rows


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="ReasonGenPilot Demo") as demo:
        gr.Markdown(
            "# ReasonGenPilot\n"
            "Unified demo for **gen** (text → image), **edit** (image + counterfactual instruction → local edit) "
            "and **hybrid** (image + counterfactual instruction → full re-paint).\n\n"
            "- Pick `auto` to let the MLLM router decide (gen if no image; edit / hybrid otherwise).\n"
            "- A single **Prompt** is required for every mode — gen reads it as a description, edit / hybrid as a counterfactual instruction.\n"
            "- Untick *Use real API* to run dry-run mode (SVG placeholders, no API spend)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Dropdown(
                    choices=["auto", "gen", "edit", "hybrid"],
                    value="auto",
                    label="Route",
                )
                prompt = gr.Textbox(
                    label="Prompt (description for gen, counterfactual for edit / hybrid)",
                    lines=3,
                )
                image = gr.Image(label="Reference image (edit / hybrid)", type="pil")
                use_real_api = gr.Checkbox(label="Use real API (uncheck = dry-run)", value=False)
                run_btn = gr.Button("Run", variant="primary")
            with gr.Column(scale=2):
                route_label = gr.Markdown("*(no run yet)*")
                with gr.Row():
                    before_img = gr.Image(label="Before", interactive=False)
                    after_img = gr.Image(label="After", interactive=False)
                reasoning = gr.Textbox(label="Reasoning & VQA", lines=14)
                result_json = gr.Code(label="Full result JSON", language="json")

        run_btn.click(
            fn=dispatch,
            inputs=[mode, prompt, image, use_real_api],
            outputs=[before_img, after_img, route_label, reasoning, result_json],
        )

        examples = _examples()
        if examples:
            gr.Examples(
                examples=examples,
                inputs=[mode, prompt, image, use_real_api],
                label="Examples",
            )

    return demo


def main() -> None:
    demo = build_demo()
    demo.queue().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )


if __name__ == "__main__":
    main()
