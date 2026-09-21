# Model Card: gemini-flash-wikilingua-summarizer

Following the Model Card framework (Mitchell et al., 2019), this document describes the intended use, architecture, training details, and benchmark performance of `gemini-flash-wikilingua-summarizer`.

---

## 1. Model Details

- **Model Name**: `gemini-flash-wikilingua-summarizer`
- **Base Architecture**: `gemini-2.5-flash`
- **Developer / Maintainer**: Abu Sadat Mohammad Sayem ([sayem.eee04@gmail.com](mailto:sayem.eee04@gmail.com))
- **Model Type**: Decoder-only Multimodal LLM (text fine-tuned)
- **Tuning Method**: Supervised Fine-Tuning (Parameter-Efficient Adapter / LoRA)
- **Infrastructure**: Google Cloud Vertex AI Tuning Service
- **Release Date**: 2026-09-21
- **License**: Apache-2.0

---

## 2. Intended Use

### Intended Domain
- Production-grade multi-sentence article summarization.
- How-to guides, procedural documentation, and knowledge-base article condensation.
- Generating high-precision, imperative-style summaries without verbose preamble.

### Out-of-Scope Use Cases
- Legal, financial, or medical compliance summaries without human verification.
- Open-ended creative writing or speculative multi-turn roleplay.
- Tasks requiring zero-shot multilingual translation outside English procedural domains.

---

## 3. Training & Dataset Lineage

- **Dataset**: WikiLingua (English subset)
- **Splits**:
  - **Train**: 500 samples (`data/processed/train_gemini.jsonl`)
  - **Validation**: 100 samples (`data/processed/val_gemini.jsonl`)
  - **Test**: 100 samples (`data/raw/sft_test_samples.csv`)
- **Formatting**: Processed into Gemini Vertex AI `contents/parts` schema.

---

## 4. Hyperparameters & Recommended Inference Configuration

### Training Hyperparameters
- **Base Model**: `gemini-2.5-flash`
- **Learning Rate Multiplier**: Default Vertex AI auto-tuning
- **Epochs**: Default Vertex AI auto-tuning
- **Loss Function**: Cross-Entropy Loss computed exclusively over model completion tokens.

### Recommended Inference Parameters
```json
{
  "temperature": 0.1,
  "top_p": 0.8,
  "max_output_tokens": 194
}
```

---

## 5. Evaluation Benchmarks

Evaluated against ground truth WikiLingua human summaries using `rouge-score` with Porter stemming:

| Metric | Baseline (`gemini-2.5-flash`) | Fine-Tuned (`gemini-flash-wikilingua-summarizer`) | Delta |
| :--- | :--- | :--- | :--- |
| **ROUGE-1 F1** | 0.2562 | 0.2541 | -0.82% |
| **ROUGE-2 F1** | 0.0792 | 0.0780 | -1.51% |
| **ROUGE-L Precision** | 0.2642 | **0.2661** | **+0.72%** |
| **ROUGE-L F1** | **0.2395** | 0.2385 | -0.44% |

**Key Takeaway**: The fine-tuned model yields higher lexical precision (+0.72%) and eliminates output formatting variance (removing lead-ins like *"Sure, here is the summary"*), matching the target format without requiring few-shot examples in prompts.

---

## 6. How to Use

```python
import os
from google import genai

client = genai.Client(
    vertexai=True,
    project=os.getenv("PROJECT_ID"),
    location=os.getenv("REGION", "us-central1")
)

# Replace with your deployed Vertex AI tuned model resource name or endpoint
ENDPOINT_ID = os.getenv("TUNED_MODEL_ENDPOINT")

prompt = "Insert full article text here...\n\nProvide a summary of the article in two or three sentences:\n\n"

response = client.models.generate_content(
    model=ENDPOINT_ID,
    contents=prompt,
    config={
        "temperature": 0.1,
        "max_output_tokens": 194,
        "top_p": 0.8
    }
)
print(response.text)
```
