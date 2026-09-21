"""
Benchmarking script to compare Baseline Gemini vs Fine-Tuned Model on test dataset.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from src.evaluation.metrics import RougeEvaluator
from src.utils.gcp import GCPManager
from src.utils.logger import get_logger

logger = get_logger("src.evaluation.benchmark")


class ModelBenchmark:
    """Runs systematic comparative evaluation across baseline and tuned endpoints."""

    def __init__(self, gcp_manager: Optional[GCPManager] = None):
        self.gcp = gcp_manager or GCPManager()
        self.client = self.gcp.get_genai_client()
        self.evaluator = RougeEvaluator()

    def generate_predictions(
        self,
        model_name_or_endpoint: str,
        prompts: List[str],
        temperature: float = 0.1,
        max_output_tokens: int = 194,
        top_p: float = 0.8,
    ) -> List[str]:
        """Generate responses for a list of input prompts."""
        logger.info(
            "Generating predictions using '%s' (%d prompts)...",
            model_name_or_endpoint,
            len(prompts),
        )
        predictions: List[str] = []

        config = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "top_p": top_p,
        }

        for prompt in tqdm(prompts, desc=f"Inference [{model_name_or_endpoint}]"):
            try:
                response = self.client.models.generate_content(
                    model=model_name_or_endpoint,
                    contents=prompt,
                    config=config,
                )
                text = response.text.strip() if response.text else ""
                predictions.append(text)
            except Exception as exc:
                logger.warning("Generation error on prompt: %s. Using empty string.", exc)
                predictions.append("")

        return predictions

    def run_benchmark(
        self,
        test_csv_path: str,
        baseline_model: str,
        tuned_endpoint: str,
        output_dir: str = "docs/artifacts",
        limit: Optional[int] = None,
        temperature: float = 0.1,
        max_output_tokens: int = 194,
    ) -> pd.DataFrame:
        """Run complete benchmark comparing baseline and fine-tuned models."""
        df = pd.read_csv(test_csv_path)
        if limit:
            df = df.head(limit)

        prompts = df["input_text"].tolist()
        references = df["output_text"].tolist()

        logger.info("Running baseline evaluation on %s...", baseline_model)
        base_preds = self.generate_predictions(
            baseline_model, prompts, temperature=temperature, max_output_tokens=max_output_tokens
        )
        base_scores = self.evaluator.evaluate_batch(references, base_preds)

        logger.info("Running fine-tuned evaluation on %s...", tuned_endpoint)
        tuned_preds = self.generate_predictions(
            tuned_endpoint, prompts, temperature=temperature, max_output_tokens=max_output_tokens
        )
        tuned_scores = self.evaluator.evaluate_batch(references, tuned_preds)

        comparison_df = self.evaluator.compare_performance(base_scores, tuned_scores)

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(out_path / "benchmark_comparison.csv", index=False)
        base_scores.to_csv(out_path / "baseline_scores.csv", index=False)
        tuned_scores.to_csv(out_path / "tuned_scores.csv", index=False)

        logger.info("Benchmark complete! Results saved to %s", output_dir)
        return comparison_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Baseline Gemini vs Tuned Model.")
    parser.add_argument(
        "--test-data", default="data/raw/sft_test_samples.csv", help="Path to test CSV."
    )
    parser.add_argument("--baseline", default="gemini-2.5-flash", help="Baseline model.")
    parser.add_argument(
        "--tuned-endpoint", required=True, help="Vertex AI tuned model endpoint or ID."
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test samples.")
    parser.add_argument(
        "--output-dir", default="docs/artifacts", help="Output directory for results."
    )
    args = parser.parse_args()

    gcp = GCPManager()
    if not gcp.verify_environment():
        logger.error("GCP environment not configured.")
        sys.exit(1)

    bench = ModelBenchmark(gcp_manager=gcp)
    comparison = bench.run_benchmark(
        test_csv_path=args.test_data,
        baseline_model=args.baseline,
        tuned_endpoint=args.tuned_endpoint,
        output_dir=args.output_dir,
        limit=args.limit,
    )
    print("\n--- BENCHMARK RESULTS ---")
    print(comparison.to_markdown(index=False))


if __name__ == "__main__":
    main()
