"""Unit tests for GeminiDataFormatter."""

import pytest

from src.data.formatter import GeminiDataFormatter


def test_convert_entry_valid():
    raw_sample = {
        "messages": [
            {"role": "user", "content": "Summarize this article."},
            {"role": "model", "content": "Here is the summary."},
        ]
    }
    result = GeminiDataFormatter.convert_entry(raw_sample)
    assert "contents" in result
    assert len(result["contents"]) == 2
    assert result["contents"][0]["role"] == "user"
    assert result["contents"][0]["parts"][0]["text"] == "Summarize this article."
    assert result["contents"][1]["role"] == "model"
    assert result["contents"][1]["parts"][0]["text"] == "Here is the summary."


def test_convert_entry_already_gemini_format():
    sample = {
        "contents": [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Hi"}]},
        ]
    }
    assert GeminiDataFormatter.convert_entry(sample) == sample


def test_convert_entry_invalid():
    with pytest.raises(ValueError):
        GeminiDataFormatter.convert_entry({"invalid_key": []})

    with pytest.raises(ValueError):
        GeminiDataFormatter.convert_entry(
            {"messages": [{"role": "unsupported_role", "content": "test"}]}
        )
