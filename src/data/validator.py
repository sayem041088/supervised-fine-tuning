"""
Dataset validation and statistics module for Gemini SFT datasets.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jsonlines

from src.utils.logger import get_logger

logger = get_logger("src.data.validator")


class DatasetValidator:
    """Validates structural correctness and token heuristics for Gemini SFT datasets."""

    @staticmethod
    def validate_entry(entry: Dict[str, Any], index: int = 0) -> Tuple[bool, List[str]]:
        """Validate a single Gemini formatted JSONL entry."""
        errors: List[str] = []

        if "contents" not in entry or not isinstance(entry["contents"], list):
            return False, [f"Row {index}: Missing 'contents' list."]

        contents = entry["contents"]
        if len(contents) < 2:
            errors.append(
                f"Row {index}: Expected at least 2 turns (user + model), got {len(contents)}."
            )

        roles = [turn.get("role") for turn in contents]
        if roles[0] != "user":
            errors.append(f"Row {index}: First turn must be 'user', got '{roles[0]}'.")

        for turn_idx, turn in enumerate(contents):
            parts = turn.get("parts")
            if not isinstance(parts, list) or not parts:
                errors.append(f"Row {index}, Turn {turn_idx}: 'parts' must be a non-empty list.")
                continue

            text = parts[0].get("text", "") if isinstance(parts[0], dict) else ""
            if not text or not str(text).strip():
                errors.append(f"Row {index}, Turn {turn_idx}: Empty text part found.")

        return len(errors) == 0, errors

    @classmethod
    def validate_file(cls, file_path: str) -> Dict[str, Any]:
        """
        Validate an entire Gemini JSONL dataset and compute basic statistics.

        Returns:
            Dictionary containing validation results and summary metrics.
        """
        p = Path(file_path)
        if not p.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        total_samples = 0
        valid_samples = 0
        all_errors: List[str] = []
        user_char_lens: List[int] = []
        model_char_lens: List[int] = []

        with jsonlines.open(p, mode="r") as reader:
            for idx, entry in enumerate(reader, start=1):
                total_samples += 1
                is_valid, errors = cls.validate_entry(entry, index=idx)
                if is_valid:
                    valid_samples += 1
                    try:
                        u_text = entry["contents"][0]["parts"][0]["text"]
                        m_text = entry["contents"][1]["parts"][0]["text"]
                        user_char_lens.append(len(u_text))
                        model_char_lens.append(len(m_text))
                    except (KeyError, IndexError):
                        pass
                else:
                    all_errors.extend(errors)

        stats = {
            "file": str(p),
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "error_count": len(all_errors),
            "errors_preview": all_errors[:5],
            "avg_prompt_chars": round(sum(user_char_lens) / len(user_char_lens), 1)
            if user_char_lens
            else 0,
            "avg_response_chars": round(sum(model_char_lens) / len(model_char_lens), 1)
            if model_char_lens
            else 0,
            "max_prompt_chars": max(user_char_lens) if user_char_lens else 0,
            "max_response_chars": max(model_char_lens) if model_char_lens else 0,
        }

        logger.info(
            "Validation finished: %d/%d valid samples. Avg prompt: %s chars, Avg target: %s chars.",
            valid_samples,
            total_samples,
            stats["avg_prompt_chars"],
            stats["avg_response_chars"],
        )
        return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Gemini SFT dataset file.")
    parser.add_argument(
        "--file", "-f", required=True, help="Path to Gemini JSONL file to validate."
    )
    args = parser.parse_args()

    results = DatasetValidator.validate_file(args.file)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
