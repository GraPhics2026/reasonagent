"""Router: dispatch user input to gen / edit / hybrid.

Single text input ``prompt`` is required for every mode — gen treats it as a
descriptive prompt, edit / hybrid treat it as a counterfactual instruction.

`auto` mode uses an MLLM classifier driven by ``prompts/router_system.txt``.
There is no fallback — if the MLLM is not configured, returns invalid JSON,
or raises any exception, this module raises ``RuntimeError`` so the caller
sees the real failure instead of a silent degradation.

Two error classes only:

- ``ValueError`` — user input is wrong (missing prompt, missing image, bad
  mode, image file not found). Caller should fix the call.
- ``RuntimeError`` — environment / system fault (MLLM unconfigured, API
  failure, malformed model response). Caller should fix the deployment.

Every message starts with ``"router: "`` so logs and tests can grep for them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from .api_client import MLLMClient, extract_json_object

Route = Literal["gen", "edit", "hybrid"]
_VALID_ROUTES: tuple[Route, ...] = ("gen", "edit", "hybrid")
_VALID_MODES: tuple[str, ...] = ("auto",) + _VALID_ROUTES

logger = logging.getLogger(__name__)

_ROUTER_PROMPT_PATH = Path("prompts/router_system.txt")


def route(
    prompt: str,
    image_path: str | Path | None = None,
    mode: str = "auto",
    client: MLLMClient | None = None,
) -> Route:
    """Decide which pipeline to dispatch to.

    Parameters
    ----------
    prompt:
        Required. User-supplied text. Acts as the description for gen and as
        the counterfactual instruction for edit / hybrid.
    image_path:
        Optional reference image. Required by edit / hybrid; ignored by gen.
    mode:
        ``"auto" | "gen" | "edit" | "hybrid"``. Anything other than ``"auto"``
        is returned verbatim after compatibility validation.
    client:
        Optional :class:`MLLMClient`. If ``None``, the router creates one and
        uses it only when ``mode == "auto"`` and an image is present.

    Returns
    -------
    One of ``"gen"`` / ``"edit"`` / ``"hybrid"``.

    Raises
    ------
    ValueError
        Prompt is empty, mode is unknown, or the chosen mode is missing a
        required input (image for edit / hybrid, or the image file does not
        exist on disk).
    RuntimeError
        ``auto`` routing needs the MLLM but it is unconfigured, the API call
        fails, or the response is invalid.
    """

    # --- 0. Prompt is required for every mode -------------------------------
    if not (prompt and prompt.strip()):
        raise ValueError("router: prompt is required for every mode.")

    # --- 1. Mode must be one of the four allowed values ---------------------
    if mode not in _VALID_MODES:
        raise ValueError(
            f"router: unknown mode {mode!r}; expected one of {_VALID_MODES}."
        )

    # --- 2. Manual mode + input compatibility -------------------------------
    if mode != "auto":
        _validate_inputs_for_mode(mode, image_path=image_path)
        return mode  # type: ignore[return-value]

    # --- 3. Auto mode + structural rules (zero cost) ------------------------
    has_image = _has_image(image_path)
    if not has_image:
        return "gen"

    # --- 4. Auto mode + MLLM classifier (image is present) ------------------
    classifier = client or MLLMClient()
    if not classifier.configured:
        raise RuntimeError(
            "router: MLLM is not configured for auto mode. "
            "Set MLLM_API_KEY and MLLM_BASE_URL in config/.env, "
            "or pass mode={'gen','edit','hybrid'} to bypass auto routing."
        )
    return _route_via_mllm(classifier, prompt.strip(), has_image=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _has_image(image_path: str | Path | None) -> bool:
    if image_path is None:
        return False
    try:
        return Path(image_path).exists()
    except (TypeError, OSError):
        return False


def _validate_inputs_for_mode(mode: str, *, image_path: str | Path | None) -> None:
    """Reject manual mode overrides whose required inputs are missing or broken."""

    if mode == "gen":
        return  # gen only needs prompt, already validated

    # edit and hybrid both need an existing image
    if not image_path:
        raise ValueError(f"router: {mode} mode requires an image.")
    if not Path(image_path).exists():
        raise ValueError(
            f"router: {mode} mode image not found at {str(image_path)!r}."
        )


def _route_via_mllm(client: MLLMClient, text: str, has_image: bool) -> Route:
    system_prompt = _ROUTER_PROMPT_PATH.read_text(encoding="utf-8")
    user_prompt = (
        f"has_image: {'yes' if has_image else 'no'}\n"
        f"user_text: {text}\n\n"
        "Return strict JSON: {\"route\": \"gen|edit|hybrid\", \"reason\": \"...\"}"
    )
    try:
        raw = client.chat_text(user_prompt, system_prompt=system_prompt, temperature=0.0)
    except Exception as exc:  # noqa: BLE001 — re-wrap with context
        raise RuntimeError(f"router: MLLM call failed ({exc}).") from exc
    try:
        data = extract_json_object(raw)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"router: MLLM returned unparseable JSON ({raw!r})."
        ) from exc
    chosen = str(data.get("route", "")).strip().lower()
    if chosen not in _VALID_ROUTES:
        raise RuntimeError(
            f"router: MLLM returned invalid route {chosen!r} (raw={raw!r})."
        )
    return chosen  # type: ignore[return-value]


__all__ = ["route", "Route"]
