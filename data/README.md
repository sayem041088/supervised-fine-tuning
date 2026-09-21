# Dataset Documentation

This directory contains the datasets used for Supervised Fine-Tuning (SFT) and evaluation of **Gemini 2.5 Flash** on article summarization.

## Source & Lineage

The dataset is derived from **[WikiLingua](https://github.com/esdurmus/Wikilingua)** (English subset), a large-scale multilingual summarization dataset extracted from WikiHow.

- **Domain**: Procedural guides, how-to articles, and step-by-step instructional texts.
- **Task**: Multi-sentence summarization capturing core actionable guidance.
- **Target Style**: Concise, imperative, bulleted or direct instructional summaries.

## Directory Layout

```text
data/
├── raw/
│   ├── sft_train_samples.jsonl   # 500 samples in standard OpenAI/Vertex chat format
│   ├── sft_val_samples.jsonl     # 100 samples for validation loss monitoring
│   └── sft_test_samples.csv      # 100 samples with input_text and ground truth output_text
└── processed/
    ├── train_gemini.jsonl        # 500 samples in Gemini contents/parts JSONL schema
    └── val_gemini.jsonl          # 100 samples in Gemini contents/parts JSONL schema
```

## Schema Specifications

### 1. Raw Format (`data/raw/*.jsonl`)
Standard multi-turn message objects:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Article text...\n\nProvide a summary of the article in two or three sentences:\n\n"
    },
    {
      "role": "model",
      "content": "Concise summary sentences."
    }
  ]
}
```

### 2. Processed Gemini SFT Format (`data/processed/*.jsonl`)
Vertex AI Gemini Supervised Fine-Tuning requires the `contents` schema with explicit `parts`:
```json
{
  "contents": [
    {
      "role": "user",
      "parts": [
        {
          "text": "Article text...\n\nProvide a summary of the article in two or three sentences:\n\n"
        }
      ]
    },
    {
      "role": "model",
      "parts": [
        {
          "text": "Concise summary sentences."
        }
      ]
    }
  ]
}
```

## Dataset Statistics

| Split | File | Sample Count | Avg Prompt Length | Avg Target Length |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | `train_gemini.jsonl` | 500 | 2,140 chars (~450 tokens) | 201 chars (~40 tokens) |
| **Val** | `val_gemini.jsonl` | 100 | 2,078 chars (~440 tokens) | 177 chars (~35 tokens) |
| **Test** | `sft_test_samples.csv` | 100 | ~2,100 chars | ~190 chars |

## Verification & Transformation

To re-transform and validate datasets:
```bash
# Format raw data to Gemini schema
make data-format

# Run validation checks
make data-validate
```
