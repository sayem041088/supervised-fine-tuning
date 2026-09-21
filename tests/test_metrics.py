"""Unit tests for RougeEvaluator."""

from src.evaluation.metrics import RougeEvaluator


def test_score_single():
    evaluator = RougeEvaluator()
    ref = "The quick brown fox jumps over the lazy dog."
    pred = "The quick brown fox jumps over a lazy dog."
    scores = evaluator.score_single(ref, pred)

    assert "rouge1_fmeasure" in scores
    assert "rouge2_fmeasure" in scores
    assert "rougeL_fmeasure" in scores
    assert 0.0 <= scores["rougeL_fmeasure"] <= 1.0
    assert scores["rouge1_fmeasure"] > 0.8


def test_evaluate_batch_and_compare():
    evaluator = RougeEvaluator()
    refs = ["Python is awesome.", "Vertex AI enables model fine tuning."]
    base_preds = ["Python is great.", "Vertex AI enables model training."]
    tuned_preds = ["Python is totally awesome.", "Vertex AI enables model fine tuning."]

    base_df = evaluator.evaluate_batch(refs, base_preds)
    tuned_df = evaluator.evaluate_batch(refs, tuned_preds)

    assert len(base_df) == 2
    assert len(tuned_df) == 2

    comparison = evaluator.compare_performance(base_df, tuned_df)
    assert len(comparison) >= 3
    assert "relative_improvement_pct" in comparison.columns
