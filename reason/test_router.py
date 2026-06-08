"""Unit tests for ``reason.router``.

Does not hit the real MLLM — all calls go through ``unittest.mock``.
Run with::

    pytest reason/test_router.py -v

Two error classes only:

- ``ValueError`` for user-input problems (missing prompt / image, bad mode,
  image file not found).
- ``RuntimeError`` for environment / system problems (MLLM unconfigured,
  API failure, malformed model response).

Every error message starts with ``"router: "`` — tests grep for that prefix.
"""

from __future__ import annotations

from unittest import mock

import pytest

from reason.router import route


# -----------------------------------------------------------------------------
# 0. Prompt is required for every mode
# -----------------------------------------------------------------------------


class TestPromptRequired:
    def test_no_prompt_raises(self):
        with pytest.raises(ValueError, match=r"router: prompt is required"):
            route(prompt="", mode="gen")

    def test_whitespace_only_prompt_raises(self):
        with pytest.raises(ValueError, match=r"router: prompt is required"):
            route(prompt="   ", mode="auto")

    def test_no_prompt_raises_for_edit(self):
        with pytest.raises(ValueError, match=r"router: prompt is required"):
            route(prompt="", image_path="README.md", mode="edit")


# -----------------------------------------------------------------------------
# 1. Mode validity
# -----------------------------------------------------------------------------


class TestModeValidity:
    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match=r"router: unknown mode"):
            route(prompt="hi", mode="bogus")

    def test_empty_string_mode_raises(self):
        with pytest.raises(ValueError, match=r"router: unknown mode"):
            route(prompt="hi", mode="")


# -----------------------------------------------------------------------------
# 2. Manual mode + input compatibility
# -----------------------------------------------------------------------------


class TestModeInputCompatibility:
    def test_gen_passthrough(self):
        assert route(prompt="A cat", mode="gen") == "gen"

    def test_edit_passthrough(self):
        assert route(prompt="if ice melts", image_path="README.md", mode="edit") == "edit"

    def test_hybrid_passthrough(self):
        assert route(prompt="变深夜", image_path="README.md", mode="hybrid") == "hybrid"

    def test_edit_without_image_raises(self):
        with pytest.raises(ValueError, match=r"router: edit mode requires an image"):
            route(prompt="if ice melts", mode="edit")

    def test_edit_with_missing_image_file_raises(self):
        with pytest.raises(ValueError, match=r"router: edit mode image not found"):
            route(prompt="x", image_path="/does/not/exist.png", mode="edit")

    def test_hybrid_without_image_raises(self):
        with pytest.raises(ValueError, match=r"router: hybrid mode requires an image"):
            route(prompt="变深夜", mode="hybrid")

    def test_hybrid_with_missing_image_file_raises(self):
        with pytest.raises(ValueError, match=r"router: hybrid mode image not found"):
            route(prompt="x", image_path="/does/not/exist.png", mode="hybrid")


# -----------------------------------------------------------------------------
# 3. Auto mode — structural rules
# -----------------------------------------------------------------------------


class TestAutoStructural:
    def test_auto_no_image_returns_gen(self):
        assert route(prompt="A cat on a window sill.", mode="auto") == "gen"

    def test_auto_nonexistent_image_returns_gen(self):
        # Treats missing file as "no image" — auto degrades to gen.
        assert route(prompt="A cat", image_path="/does/not/exist.png", mode="auto") == "gen"


# -----------------------------------------------------------------------------
# 4. Auto mode — MLLM happy path
# -----------------------------------------------------------------------------


def _make_mllm(response: str):
    client = mock.MagicMock()
    client.configured = True
    client.chat_text.return_value = response
    return client


class TestMLLMPath:
    def test_mllm_returns_edit(self):
        client = _make_mllm('{"route": "edit", "reason": "local change"}')
        assert (
            route(prompt="如果冰块融化", image_path="README.md", mode="auto", client=client)
            == "edit"
        )

    def test_mllm_returns_hybrid(self):
        client = _make_mllm('{"route": "hybrid", "reason": "full repaint"}')
        assert (
            route(prompt="变深夜", image_path="README.md", mode="auto", client=client)
            == "hybrid"
        )

    def test_mllm_returns_gen(self):
        client = _make_mllm('{"route": "gen", "reason": "..."}')
        assert (
            route(prompt="x", image_path="README.md", mode="auto", client=client) == "gen"
        )

    def test_mllm_handles_markdown_fenced_json(self):
        client = _make_mllm('```json\n{"route": "edit", "reason": "x"}\n```')
        assert (
            route(prompt="if ice melts", image_path="README.md", mode="auto", client=client)
            == "edit"
        )


# -----------------------------------------------------------------------------
# 5. Auto mode — MLLM failures (no fallback, must raise RuntimeError)
# -----------------------------------------------------------------------------


class TestMLLMErrors:
    def test_mllm_unconfigured_raises(self, monkeypatch):
        class _DummyClient:
            configured = False

            def chat_text(self, *args, **kwargs):  # pragma: no cover
                raise AssertionError("MLLM should not be called when unconfigured")

        monkeypatch.setattr("reason.router.MLLMClient", _DummyClient)
        with pytest.raises(RuntimeError, match=r"router: MLLM is not configured"):
            route(prompt="如果冰块融化", image_path="README.md", mode="auto")

    def test_mllm_invalid_route_value_raises(self):
        client = _make_mllm('{"route": "garbage", "reason": "..."}')
        with pytest.raises(RuntimeError, match=r"router: MLLM returned invalid route"):
            route(prompt="x", image_path="README.md", mode="auto", client=client)

    def test_mllm_unparseable_response_raises(self):
        client = _make_mllm("not json at all")
        with pytest.raises(RuntimeError, match=r"router: MLLM returned unparseable JSON"):
            route(prompt="x", image_path="README.md", mode="auto", client=client)

    def test_mllm_api_exception_raises(self):
        client = mock.MagicMock()
        client.configured = True
        client.chat_text.side_effect = RuntimeError("network down")
        with pytest.raises(RuntimeError, match=r"router: MLLM call failed"):
            route(prompt="x", image_path="README.md", mode="auto", client=client)

    def test_mllm_missing_route_key_raises(self):
        client = _make_mllm('{"reason": "no route key"}')
        with pytest.raises(RuntimeError, match=r"router: MLLM returned invalid route"):
            route(prompt="x", image_path="README.md", mode="auto", client=client)
