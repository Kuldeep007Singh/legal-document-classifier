# Code Path: src/inference/predict.py

import json
import torch
import logging
from pathlib import Path
from typing import Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_DIR  = Path("models/best_models/legal_bert_doc_classifier")
MAX_LENGTH = 512

# Document type descriptions shown to the user
DOC_DESCRIPTIONS = {
    "scotus":     "US Supreme Court Ruling",
    "ledgar":     "Contract / Legal Agreement Clause",
    "eurlex":     "European Union Regulation / Legislation",
    "ecthr_a":    "European Court of Human Rights Case (Violation Articles)",
    "ecthr_b":    "European Court of Human Rights Case (Alleged Violations)",
    "case_hold":  "Legal Case Holding / Precedent",
    "unfair_tos": "Terms of Service Clause (Unfairness Detection)",
}


class LegalDocumentClassifier:
    """
    Classifies legal documents into one of 7 types using fine-tuned Legal-BERT.

    Supported document types:
        scotus, ledgar, eurlex, ecthr_a, ecthr_b, case_hold, unfair_tos
    """

    def __init__(self, model_dir: Union[str, Path] = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model     = None
        self.tokenizer = None
        self.label_map = None
        self._load()

    def _load(self):
        """Load model, tokenizer and label map from disk."""
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_dir}. "
                "Download it from Colab and place it in models/best_models/legal_bert_doc_classifier/"
            )

        logger.info(f"Loading model from {self.model_dir} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model     = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()

        label_map_path = self.model_dir / "label_map.json"
        with open(label_map_path, "r") as f:
            raw = json.load(f)
        # keys are stored as strings in JSON, convert to int
        self.label_map = {int(k): v for k, v in raw.items()}

        logger.info(f"Model loaded. Classes: {list(self.label_map.values())}")

    def predict(self, text: str) -> dict:
        """
        Predict the document type for a given text.

        Args:
            text: Raw legal document text (will be truncated to 512 tokens)

        Returns:
            dict with keys:
                - predicted_class: short label e.g. 'scotus'
                - description:     human-readable document type
                - confidence:      float 0-1
                - all_scores:      dict of {class: confidence} for all 7 classes
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs      = torch.softmax(outputs.logits, dim=-1).squeeze()
        pred_id    = int(torch.argmax(probs).item())
        pred_class = self.label_map[pred_id]
        confidence = float(probs[pred_id].item())

        all_scores = {
            self.label_map[i]: round(float(probs[i].item()), 4)
            for i in range(len(self.label_map))
        }
        # sort by confidence descending
        all_scores = dict(sorted(all_scores.items(), key=lambda x: x[1], reverse=True))

        return {
            "predicted_class": pred_class,
            "description":     DOC_DESCRIPTIONS.get(pred_class, pred_class),
            "confidence":      round(confidence, 4),
            "all_scores":      all_scores,
        }

    def predict_batch(self, texts: list) -> list:
        """
        Predict document types for a list of texts.

        Args:
            texts: List of raw legal document strings

        Returns:
            List of prediction dicts (same format as predict())
        """
        return [self.predict(text) for text in texts]


# ── Standalone usage ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    classifier = LegalDocumentClassifier()

    # Demo texts — one per document type
    demo_texts = [
        (
            "scotus",
            "The Court held that the Fourth Amendment prohibition on unreasonable searches "
            "applies to digital data stored on cell phones seized incident to arrest. "
            "Riley v. California, 573 U.S. 373 (2014)."
        ),
        (
            "ledgar",
            "The parties agree that this Agreement shall be governed by and construed in "
            "accordance with the laws of the State of Delaware, without regard to its "
            "conflict of law provisions."
        ),
        (
            "eurlex",
            "The Commission, having regard to the Treaty on the Functioning of the European "
            "Union, and in particular Article 108(2) thereof, hereby decides that the state "
            "aid granted by the Member State is incompatible with the internal market."
        ),
        (
            "unfair_tos",
            "We reserve the right to modify or discontinue the service at any time without "
            "notice. You agree that we shall not be liable to you or any third party for any "
            "modification, suspension, or discontinuation of the service."
        ),
    ]

    print("\n" + "=" * 65)
    print(" Legal Document Classifier — Inference Demo")
    print("=" * 65)

    all_correct = 0
    for expected, text in demo_texts:
        result = classifier.predict(text)
        status = "CORRECT" if result["predicted_class"] == expected else "WRONG"
        if status == "CORRECT":
            all_correct += 1

        print(f"\n[{status}]")
        print(f"  Expected  : {expected}")
        print(f"  Predicted : {result['predicted_class']} — {result['description']}")
        print(f"  Confidence: {result['confidence']*100:.1f}%")
        print(f"  Top scores: ", end="")
        top3 = list(result["all_scores"].items())[:3]
        print(" | ".join(f"{k}: {v*100:.1f}%" for k, v in top3))

    print(f"\n{'='*65}")
    print(f"Demo accuracy: {all_correct}/{len(demo_texts)}")
    print("=" * 65)

    # Interactive mode if text passed as argument
    if len(sys.argv) > 1:
        user_text = " ".join(sys.argv[1:])
        print(f"\nRunning inference on provided text...")
        result = classifier.predict(user_text)
        print(f"Predicted: {result['predicted_class']} — {result['description']}")
        print(f"Confidence: {result['confidence']*100:.1f}%")