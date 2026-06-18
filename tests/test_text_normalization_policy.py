from __future__ import annotations

from vlm2emb.data.datasets.const import canonicalize_multimodal_text, normalize_text_whitespace


def test_normalize_text_whitespace_no_longer_adds_trailing_newline_by_default():
    assert normalize_text_whitespace("hello") == "hello"
    assert normalize_text_whitespace(" hello  ") == "hello"
    assert normalize_text_whitespace("hello\n") == "hello\n"


def test_normalize_text_whitespace_preserves_existing_trailing_newlines_by_default():
    assert normalize_text_whitespace("hello\n\n") == "hello\n\n"


def test_normalize_text_whitespace_can_still_add_trailing_newline_explicitly():
    assert normalize_text_whitespace("hello", ensure_trailing_newline=True) == "hello\n"


def test_canonicalize_multimodal_text_no_longer_adds_trailing_newline_by_default():
    assert canonicalize_multimodal_text("<|image_pad|>\nhello", token="<|image_pad|>") == "<|image_pad|>\nhello"


def test_canonicalize_multimodal_text_preserves_existing_trailing_newline_by_default():
    assert canonicalize_multimodal_text("<|image_pad|>\nhello\n", token="<|image_pad|>") == "<|image_pad|>\nhello\n"


def test_canonicalize_multimodal_text_keeps_explicit_trailing_newline_opt_in():
    assert canonicalize_multimodal_text(
        "<|image_pad|>\nhello",
        token="<|image_pad|>",
        ensure_trailing_newline=True,
    ) == "<|image_pad|>\nhello\n"
