"""Unit tests for DatasetValidator."""

from src.data.validator import DatasetValidator


def test_validate_entry_valid():
    valid_entry = {
        "contents": [
            {"role": "user", "parts": [{"text": "Valid question"}]},
            {"role": "model", "parts": [{"text": "Valid answer"}]},
        ]
    }
    is_valid, errors = DatasetValidator.validate_entry(valid_entry, index=1)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_entry_empty_parts():
    invalid_entry = {
        "contents": [
            {"role": "user", "parts": []},
            {"role": "model", "parts": [{"text": "Answer"}]},
        ]
    }
    is_valid, errors = DatasetValidator.validate_entry(invalid_entry, index=1)
    assert is_valid is False
    assert any("'parts' must be a non-empty list" in err for err in errors)


def test_validate_entry_invalid_first_turn():
    invalid_entry = {
        "contents": [
            {"role": "model", "parts": [{"text": "Answer first"}]},
            {"role": "user", "parts": [{"text": "Question"}]},
        ]
    }
    is_valid, errors = DatasetValidator.validate_entry(invalid_entry, index=1)
    assert is_valid is False
    assert any("First turn must be 'user'" in err for err in errors)
