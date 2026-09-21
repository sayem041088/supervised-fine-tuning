"""
Data formatting module for Gemini Supervised Fine-Tuning.
Converts conversational datasets into Vertex AI Gemini SFT JSONL format.
"""

import argparse
from pathlib import Path
from typing import Any, Dict

import jsonlines

from src.utils.logger import get_logger

logger = get_logger("src.data.formatter")


class GeminiDataFormatter:
    """Formats chat messages into Gemini 2.5 contents/parts JSONL schema."""

    @staticmethod
    def convert_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a single chat-formatted entry to Gemini SFT format.

        Input schema:
            {"messages": [{"role": "user", "content": "..."}, {"role": "model", "content": "..."}]}
        Output schema:
            {"contents": [{"role": "user", "parts": [{"text": "..."}]}, {"role": "model", "parts": [{"text": "..."}]}]}
        """
        if "contents" in entry:
            return entry

        if "messages" not in entry or not isinstance(entry["messages"], list):
            raise ValueError("Input entry must contain a 'messages' list.")

        gemini_contents = []
        for msg in entry["messages"]:
            role = msg.get("role")
            content = msg.get("content", "")
            if not role or role not in ("user", "model", "system"):
                raise ValueError(
                    f"Invalid message role '{role}'. Expected 'user', 'model', or 'system'."
                )

            gemini_contents.append({"role": role, "parts": [{"text": content}]})

        return {"contents": gemini_contents}

    @classmethod
    def process_file(cls, input_path: str, output_path: str) -> int:
        """
        Process a full JSONL file and write formatted Gemini entries.

        Args:
            input_path: Path to source JSONL file.
            output_path: Path to destination JSONL file.

        Returns:
            Number of successfully converted samples.
        """
        in_p = Path(input_path)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if not in_p.is_file():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        count = 0
        logger.info("Processing %s -> %s...", input_path, output_path)
        with jsonlines.open(in_p, mode="r") as reader, jsonlines.open(out_p, mode="w") as writer:
            for line_no, sample in enumerate(reader, start=1):
                try:
                    formatted = cls.convert_entry(sample)
                    writer.write(formatted)
                    count += 1
                except Exception as exc:
                    logger.warning(
                        "Skipping invalid entry at line %d in %s: %s", line_no, input_path, exc
                    )

        logger.info("Successfully formatted %d records into %s", count, output_path)
        return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert chat JSONL datasets to Gemini SFT format."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input JSONL dataset.")
    parser.add_argument("--output", "-o", required=True, help="Path to output JSONL file.")
    args = parser.parse_args()

    GeminiDataFormatter.process_file(args.input, args.output)


if __name__ == "__main__":
    main()
