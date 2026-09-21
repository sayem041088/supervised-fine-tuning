"""
Evaluation metrics module for summarization using ROUGE scores.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
from rouge_score import rouge_scorer

from src.utils.logger import get_logger

logger = get_logger("src.evaluation.metrics")


class RougeEvaluator:
    """Evaluates generated summaries against references using ROUGE metrics."""

    def __init__(self, metrics: Optional[List[str]] = None, use_stemmer: bool = True):
        self.metric_names = metrics or ["rouge1", "rouge2", "rougeL"]
        self.scorer = rouge_scorer.RougeScorer(self.metric_names, use_stemmer=use_stemmer)

    def score_single(self, reference: str, prediction: str) -> Dict[str, float]:
        """
        Compute ROUGE scores for a single pair of reference and prediction.

        Returns:
            Dictionary mapping metric_name (e.g. rouge1_fmeasure) to float value.
        """
        raw_scores = self.scorer.score(reference, prediction)
        flattened: Dict[str, float] = {}
        for name, score in raw_scores.items():
            flattened[f"{name}_precision"] = score.precision
            flattened[f"{name}_recall"] = score.recall
            flattened[f"{name}_fmeasure"] = score.fmeasure
        return flattened

    def evaluate_batch(
        self,
        references: List[str],
        predictions: List[str],
        indices: Optional[List[Any]] = None,
    ) -> pd.DataFrame:
        """
        Compute ROUGE scores for a batch of predictions and references.

        Returns:
            DataFrame containing individual sample scores and summary statistics.
        """
        if len(references) != len(predictions):
            raise ValueError(
                f"Mismatch: {len(references)} references vs {len(predictions)} predictions."
            )

        records: List[Dict[str, Any]] = []
        for idx, (ref, pred) in enumerate(zip(references, predictions, strict=True)):
            sample_id = indices[idx] if indices is not None else idx
            scores = self.score_single(reference=ref, prediction=pred)
            scores["sample_id"] = sample_id
            scores["ref_length"] = len(ref)
            scores["pred_length"] = len(pred)
            records.append(scores)

        df = pd.DataFrame(records)
        logger.info(
            "Batch evaluation complete (%d samples). Mean ROUGE-L F1: %.4f",
            len(df),
            df["rougeL_fmeasure"].mean() if "rougeL_fmeasure" in df else 0.0,
        )
        return df

    @staticmethod
    def compare_performance(
        baseline_df: pd.DataFrame,
        tuned_df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compare baseline and fine-tuned performance across ROUGE metrics.

        Returns:
            DataFrame with Baseline, Tuned, Absolute Delta, and Improvement (%).
        """
        metrics = metrics or ["rouge1_fmeasure", "rouge2_fmeasure", "rougeL_fmeasure"]
        comparison_rows = []

        for m in metrics:
            if m not in baseline_df.columns or m not in tuned_df.columns:
                continue

            base_mean = float(baseline_df[m].mean())
            tuned_mean = float(tuned_df[m].mean())
            abs_delta = tuned_mean - base_mean
            pct_improvement = (abs_delta / base_mean * 100.0) if base_mean != 0 else 0.0

            comparison_rows.append(
                {
                    "metric": m,
                    "baseline_mean": base_mean,
                    "tuned_mean": tuned_mean,
                    "absolute_delta": abs_delta,
                    "relative_improvement_pct": pct_improvement,
                }
            )

        return pd.DataFrame(comparison_rows)
