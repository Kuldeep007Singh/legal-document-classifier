# Legal Document Classifier

A Legal-BERT based document classification system trained on the LexGLUE benchmark. Given any legal text, it identifies which type of legal document it belongs to — Supreme Court opinion, EU regulation, contract clause, human rights case, or terms of service.

---

## Why I Built This

Legal documents are dense and varied. A SCOTUS opinion looks nothing like an EU regulation, and a contract clause is completely different from a human rights case. I wanted to build something that could automatically identify what kind of legal document it's dealing with — as a first step toward building more specialized legal NLP pipelines.

I chose Legal-BERT (`nlpaueb/legal-bert-base-uncased`) over standard BERT because it was pre-trained on 12GB of legal corpora — court cases, EU legislation, contracts — which makes it far more aware of legal terminology and citation patterns than a general-purpose model.

---

## What It Does

Takes any legal document text as input and classifies it into one of 7 categories:

| Label | Document Type |
|---|---|
| `scotus` | US Supreme Court Ruling |
| `ledgar` | Contract / Legal Agreement Clause |
| `eurlex` | European Union Regulation / Legislation |
| `ecthr_a` | ECtHR Case — Violation Articles |
| `ecthr_b` | ECtHR Case — Alleged Violations |
| `case_hold` | Legal Case Holding / Precedent |
| `unfair_tos` | Terms of Service Clause |

---

## Dataset

I used the [LexGLUE benchmark](https://github.com/coastalcph/lex-glue) — 7 separate legal NLP datasets. Instead of training on each dataset independently, I combined them into a single balanced classification task where the label is the document type (which dataset it came from).

- **35,000 training samples** — 5,000 per class, perfectly balanced
- **7,000 validation samples** — 1,000 per class
- **7,000 test samples** — 1,000 per class
- All splits use the original LexGLUE train/val/test files — no leakage

The sampling and combining script is in `notebooks/legal_doc_classifier.ipynb`.

---

## Model Performance

| Metric | Score |
|---|---|
| Test Accuracy | 85.03% |
| Macro F1 | 84.66% |
| Weighted F1 | 84.66% |

**Per-class breakdown:**

| Class | F1 Score |
|---|---|
| eurlex | 1.00 |
| ledgar | 0.99 |
| scotus | 0.98 |
| case_hold | 0.98 |
| unfair_tos | 0.99 |
| ecthr_a | 0.57 |
| ecthr_b | 0.41 |

Five out of seven classes hit 98–100% F1. The overall number is pulled down by `ecthr_a` and `ecthr_b` — these are the same court documents labelled differently based on violation type, so the model has no writing-style signal to separate them. This is a dataset-level ambiguity, not a model failure.

---

## Project Structure

```
legal-document-classifier/
├── data/
│   ├── raw/                        # Original LexGLUE CSVs (7 datasets × 3 splits)
│   └── processed/                  # Combined balanced CSVs used for training
├── models/
│   └── best_models/
│       ├── legal_bert_doc_classifier/   # Fine-tuned Legal-BERT weights
│       └── traditional_*.joblib         # TF-IDF + RandomForest baseline
├── notebooks/
│   └── legal_doc_classifier.ipynb  # Full training notebook (run on Colab T4)
├── src/
│   ├── data/                       # Data loading and preprocessing
│   ├── models/                     # Model architectures
│   ├── training/                   # Trainer, callbacks, validator
│   ├── inference/
│   │   └── predict.py              # Main inference class — use this for predictions
│   ├── evaluation/                 # Metrics
│   └── api/                        # FastAPI backend
├── ui/
│   └── streamlit_app.py            # Streamlit UI
├── requirements.txt
└── Dockerfile
```

---

## Setup

```bash
git clone https://github.com/your-username/legal-document-classifier
cd legal-document-classifier
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Download the model weights and place them in `models/best_models/legal_bert_doc_classifier/`. The folder should contain: `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `label_map.json`.

---

## Usage

**Run inference directly:**

```python
from src.inference.predict import LegalDocumentClassifier

clf = LegalDocumentClassifier()

result = clf.predict("""
    The Court held that the Fourth Amendment prohibition on unreasonable searches
    applies to digital data stored on cell phones seized incident to arrest.
""")

print(result["predicted_class"])   # scotus
print(result["confidence"])        # 0.9934
print(result["description"])       # US Supreme Court Ruling
print(result["all_scores"])        # scores for all 7 classes
```

**Run the Streamlit UI:**

```bash
python -m streamlit run ui/streamlit_app.py
```

**Run the FastAPI backend:**

```bash
uvicorn src.api.app:app --reload
```

---

## Training

Training was done on Google Colab with a T4 GPU. The full notebook is at `notebooks/legal_doc_classifier.ipynb`.

Key training config:
- Model: `nlpaueb/legal-bert-base-uncased`
- Max sequence length: 512 tokens
- Batch size: 8 (with gradient accumulation steps = 4, effective batch = 32)
- Learning rate: 2e-5
- Epochs: 3
- Mixed precision: fp16
- Early stopping patience: 2

---

## Limitations and Next Steps

The current model is a **document-type classifier** — it tells you what kind of legal document something is, not what it's about. The natural next step is a two-stage pipeline:

1. **Stage 1 (this project):** Identify document type
2. **Stage 2 (future work):** Run a document-type-specific classifier — e.g., for SCOTUS opinions, predict the area of law; for contracts, detect clause types; for ToS, detect unfair clauses

Stage 2 would require handling multi-label classification properly (a single clause can violate multiple ToS categories simultaneously), which needs `BCEWithLogitsLoss` instead of `CrossEntropyLoss` and a different evaluation strategy.

---

## Tech Stack

Python · PyTorch · HuggingFace Transformers · Legal-BERT · scikit-learn · MLflow · FastAPI · Streamlit · Docker