"""retrain_model.py

Automated fine-tuning workflow for a PII detection model (spaCy-backed).

This script is intentionally defensive: it checks for spaCy and falls back
with a helpful error message if the environment lacks the required libraries.

Expected correction dataset format: JSONL with one object per line:
  {"text": "...", "entities": [[start, end, "LABEL"], ...]}

Validation dataset uses the same format.

Usage:
  python3 scripts/retrain_model.py --prod-model ./models/prod --corrections data/corrections.jsonl --validation data/validation.jsonl --epochs 2 --out models --tag pii-detector-v1.2.1
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _require_spacy():
    try:
        import spacy  # type: ignore
        from spacy.tokens import DocBin  # type: ignore

        return spacy, DocBin
    except Exception as exc:  # pragma: no cover - environment may not have spacy
        raise RuntimeError(
            "spaCy is required to run this script. Install with `pip install spacy` "
            "and a compatible model (e.g., `python -m spacy download en_core_web_sm`)."
        ) from exc


def load_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def prepare_training_data(corrections_path: Path) -> List[Tuple[str, Dict]]:
    """Return spaCy-compatible training examples: (text, {'entities': [...]})"""
    examples = []
    for obj in load_jsonl(corrections_path):
        text = obj.get("text")
        ents = obj.get("entities") or []
        # ensure entities are tuples of (start, end, label)
        normalized = []
        for e in ents:
            if len(e) >= 3:
                normalized.append((int(e[0]), int(e[1]), str(e[2])))
        examples.append((text, {"entities": normalized}))
    return examples


def evaluate_model(nlp, validation_path: Path) -> Dict[str, float]:
    """Evaluate model on validation set and return precision/recall/f1 for entities.

    Uses exact match on spans+labels. Returns aggregate scores.
    """
    tp = 0
    fp = 0
    fn = 0
    total_ents = 0
    for obj in load_jsonl(validation_path):
        text = obj.get("text", "")
        gold_ents = {(int(a), int(b), str(l)) for a, b, l in obj.get("entities", [])}
        total_ents += len(gold_ents)
        doc = nlp(text)
        pred_ents = {(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents}
        for p in pred_ents:
            if p in gold_ents:
                tp += 1
            else:
                fp += 1
        for g in gold_ents:
            if g not in pred_ents:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "total_entities": total_ents}


def fine_tune_spacy_model(
    prod_model_path: Path,
    corrections: List[Tuple[str, Dict]],
    epochs: int = 2,
    dropout: float = 0.2,
    seed: int | None = None,
    learning_rate: float | None = None,
    batch_size: int | None = None,
):
    """Fine-tune a spaCy model with optional learning_rate and batch_size.

    This function uses spaCy's `nlp.update` loop with minibatches. We attempt
    to set the optimizer learning rate when possible (older spaCy optimizers
    expose an `alpha` attribute).
    """
    spacy, DocBin = _require_spacy()
    random.seed(seed)
    # Load production NLP
    nlp = spacy.load(str(prod_model_path))

    # Get the NER component (create if missing)
    if "ner" not in nlp.pipe_names:
        # Create and add a new ner component
        ner = nlp.add_pipe("ner")  # type: ignore
    else:
        ner = nlp.get_pipe("ner")

    # Add labels from corrections into the NER's label set
    labels = set()
    for _, ann in corrections:
        for s, e, lab in ann.get("entities", []):
            labels.add(lab)
    for lab in labels:
        try:
            ner.add_label(lab)
        except Exception:
            # label may already exist
            pass

    # Prepare optimizer
    optimizer = nlp.resume_training()
    if learning_rate is not None:
        try:
            # Some spaCy optimizer variants accept an `alpha` attribute
            setattr(optimizer, "alpha", float(learning_rate))
        except Exception:
            # Not all optimizer types accept this; ignore if not supported
            pass

    # Use spaCy minibatch utility if available to support batch_size
    try:
        from spacy.util import minibatch  # type: ignore
    except Exception:
        minibatch = None  # type: ignore

    for i in range(epochs):
        random.shuffle(corrections)
        losses = {}
        if minibatch and batch_size:
            # create batches of (text, ann)
            batches = minibatch(corrections, size=batch_size)
            for batch in batches:
                texts = [t for t, a in batch]
                anns = [a for t, a in batch]
                try:
                    nlp.update(texts, anns, sgd=optimizer, drop=dropout, losses=losses)
                except Exception:
                    continue
        else:
            for text, ann in corrections:
                try:
                    nlp.update([text], [ann], sgd=optimizer, drop=dropout, losses=losses)
                except Exception:
                    continue
        logging.info("Epoch %d finished. Losses: %s", i + 1, losses)

    return nlp


def tune_hyperparameters(
    prod_model_path: Path,
    corrections_path: Path,
    validation_path: Path,
    n_trials: int = 50,
    seed: int = 42,
) -> Dict:
    """Run hyperparameter tuning using Optuna.

    The objective trains a fresh copy of the production model with trial-suggested
    hyperparameters (learning_rate, batch_size, dropout, epochs) and returns the
    validation F1. The study maximizes F1 across `n_trials`.
    """
    try:
        import optuna  # type: ignore
    except Exception as exc:
        raise RuntimeError("Optuna is required for hyperparameter tuning. Install with `pip install optuna`") from exc

    corrections = prepare_training_data(corrections_path)
    if not corrections:
        raise RuntimeError("No correction examples available for tuning")

    def objective(trial: "optuna.trial.Trial") -> float:  # type: ignore
        # Suggest hyperparameters
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
        batch_size = int(trial.suggest_categorical("batch_size", [8, 16, 32, 64]))
        dropout = float(trial.suggest_float("dropout", 0.1, 0.5))
        epochs = int(trial.suggest_int("epochs", 1, 3))

        # Fine-tune a fresh copy of prod model
        tuned_nlp = fine_tune_spacy_model(
            prod_model_path,
            corrections,
            epochs=epochs,
            dropout=dropout,
            seed=seed,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

        scores = evaluate_model(tuned_nlp, validation_path)
        f1 = float(scores.get("f1", 0.0))
        # Log trial details
        trial.set_user_attr("scores", scores)
        logging.info("Trial done: lr=%s bs=%s drop=%s epochs=%s -> f1=%.4f", learning_rate, batch_size, dropout, epochs, f1)
        return f1

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    logging.info("Best trial: value=%s params=%s", best.value, best.params)
    return {"best_value": best.value, "best_params": best.params}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrain/fine-tune PII detection model using correction dataset")
    p.add_argument("--prod-model", required=True, help="Path to the production model directory (spaCy)")
    p.add_argument("--corrections", required=True, help="Path to corrections JSONL file")
    p.add_argument("--validation", required=True, help="Path to validation JSONL file")
    p.add_argument("--epochs", type=int, default=2, help="Fine-tuning epochs (1-3 recommended)")
    p.add_argument("--out", required=True, help="Output directory to save new model if improved")
    p.add_argument("--tag", required=True, help="Version tag for the new model (e.g., pii-detector-v1.2.1)")
    p.add_argument("--dropout", type=float, default=0.2, help="Dropout rate during training")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    prod_model_path = Path(args.prod_model)
    corrections_path = Path(args.corrections)
    validation_path = Path(args.validation)
    out_dir = Path(args.out)

    if not prod_model_path.exists():
        logging.error("Production model path not found: %s", prod_model_path)
        return 2
    if not corrections_path.exists():
        logging.error("Corrections file not found: %s", corrections_path)
        return 2
    if not validation_path.exists():
        logging.error("Validation file not found: %s", validation_path)
        return 2

    # Load baseline production model for evaluation
    try:
        spacy, _ = _require_spacy()
        prod_nlp = spacy.load(str(prod_model_path))
    except Exception as exc:
        logging.exception("Failed to load production model: %s", exc)
        return 3

    # Evaluate prod model on validation
    logging.info("Evaluating production model on validation set...")
    prod_scores = evaluate_model(prod_nlp, validation_path)
    logging.info("Production model scores: %s", prod_scores)

    # Load corrections and prepare training pairs
    corrections = prepare_training_data(corrections_path)
    if not corrections:
        logging.error("No correction examples found; aborting")
        return 4

    # Fine-tune
    logging.info("Starting fine-tuning: epochs=%d", args.epochs)
    tuned_nlp = fine_tune_spacy_model(prod_model_path, corrections, epochs=args.epochs, dropout=args.dropout, seed=args.seed)

    # Evaluate tuned model
    logging.info("Evaluating tuned model on validation set...")
    tuned_scores = evaluate_model(tuned_nlp, validation_path)
    logging.info("Tuned model scores: %s", tuned_scores)

    # Compare F1 and decide to save
    prod_f1 = prod_scores.get("f1", 0.0)
    tuned_f1 = tuned_scores.get("f1", 0.0)
    logging.info("Prod F1=%.4f Tuned F1=%.4f", prod_f1, tuned_f1)

    if tuned_f1 > prod_f1:
        # Save new model
        target = out_dir / args.tag
        target.mkdir(parents=True, exist_ok=True)
        tuned_nlp.to_disk(str(target))
        logging.info("New model saved to: %s", target)
        print(json.dumps({"saved": True, "path": str(target), "prod_f1": prod_f1, "tuned_f1": tuned_f1}))
        return 0
    else:
        logging.warning("Tuned model did not improve F1. Not saving. Prod F1=%.4f Tuned F1=%.4f", prod_f1, tuned_f1)
        print(json.dumps({"saved": False, "prod_f1": prod_f1, "tuned_f1": tuned_f1}))
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
