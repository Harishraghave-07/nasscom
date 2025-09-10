#!/usr/bin/env python3
"""Generate synthetic ground-truth documents containing India-specific PII.

This script creates realistic-looking sentences using Faker (locale 'en_IN')
and injects India-specific PII into the text. It writes N text files and a
`ground_truth.jsonl` file containing annotations with exact character
offsets for evaluation.

Generation logic
----------------
- Build a paragraph by concatenating short sentences from Faker.
- With probability `pii_density` each sentence will have one embedded PII
  token (picked from critical / high-priority lists).
- When embedding a PII token we append it to the sentence; we record the
  document offset (start/end) at insertion time which guarantees exact
  character positions in the produced text.

Usage
-----
python scripts/generate_test_data.py

Requirements
------------
pip install faker

"""

from __future__ import annotations

import json
import os
import random
import string
from pathlib import Path
from typing import Dict, List, Tuple

from faker import Faker


fake = Faker("en_IN")


def _random_digits(n: int) -> str:
    return "".join(random.choice(string.digits) for _ in range(n))


def gen_aadhaar() -> str:
    # Format 4-4-4 for readability
    parts = [_random_digits(4) for _ in range(3)]
    return "-".join(parts)


def gen_pan() -> str:
    # 5 letters, 4 digits, 1 letter
    letters = "".join(random.choice(string.ascii_uppercase) for _ in range(5))
    digits = _random_digits(4)
    last = random.choice(string.ascii_uppercase)
    return f"{letters}{digits}{last}"


def gen_bank_account() -> str:
    # Typical Indian bank accounts vary; generate 9-18 digits
    n = random.randint(9, 18)
    return _random_digits(n)


def gen_credit_card() -> str:
    # Simple 16-digit formatted CC
    parts = [_random_digits(4) for _ in range(4)]
    return " ".join(parts)


def gen_mobile() -> str:
    # +91 followed by 10 digits, start with 7/8/9
    first = random.choice("789")
    rest = _random_digits(9)
    return f"+91-{first}{rest[:4]}-{rest[4:]}"


def gen_pin_code() -> str:
    return _random_digits(6)


def gen_address() -> str:
    # Use faker address and append a PIN code to make it India-like
    addr = fake.address().replace("\n", ", ")
    pin = gen_pin_code()
    return f"{addr}, PIN: {pin}"


PII_GENERATORS = [
    ("AADHAAR_NUMBER", gen_aadhaar),
    ("PAN_NUMBER", gen_pan),
    ("BANK_ACCOUNT", gen_bank_account),
    ("CREDIT_CARD", gen_credit_card),
    ("MOBILE_NUMBER", gen_mobile),
    ("NAME", lambda: fake.name()),
    ("ADDRESS", gen_address),
    ("EMAIL_ADDRESS", lambda: fake.email()),
]


def generate_synthetic_document(pii_density: float = 0.3, seed: int | None = None) -> Tuple[str, List[Dict]]:
    """Generate a synthetic document string and annotations.

    Args:
        pii_density: probability that a sentence contains one PII token.
        seed: optional random seed for reproducibility.

    Returns:
        (text, annotations)
        - text: generated document string
        - annotations: list of dicts {text, entity, start, end}
    """
    if seed is not None:
        random.seed(seed)

    sentences = []
    annotations: List[Dict] = []

    # produce a variable number of sentences per document
    n_sentences = random.randint(6, 16)
    doc = ""

    for _ in range(n_sentences):
        s = fake.sentence(nb_words=random.randint(6, 14))
        # Decide whether to embed a PII token
        if random.random() < pii_density:
            ent_name, gen = random.choice(PII_GENERATORS)
            pii_value = gen()
            # Append the PII token to the sentence with surrounding punctuation
            insert = f" {pii_value}"
            start = len(doc) + len(s) + 1  # +1 for the space we'll add below
            end = start + len(pii_value)
            annotations.append({"text": pii_value, "entity": ent_name, "start": start, "end": end})
            s = s.rstrip(".") + "." + insert

        # append sentence to doc with a space separator
        if doc:
            doc += " " + s
        else:
            doc = s

    # final housekeeping: normalize whitespace
    doc = "\n\n".join([p.strip() for p in doc.split("\n\n")])
    return doc, annotations


def main(out_dir: str = "./synthetic_data", n: int = 100, pii_density: float = 0.35) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    gt_path = out_path / "ground_truth.jsonl"
    with open(gt_path, "w", encoding="utf-8") as gt_fh:
        for i in range(n):
            doc_text, annotations = generate_synthetic_document(pii_density=pii_density)
            file_name = f"sample_{i:04d}.txt"
            with open(out_path / file_name, "w", encoding="utf-8") as fh:
                fh.write(doc_text)

            # write one JSONL line per document with annotations
            gt_fh.write(json.dumps({"file": file_name, "text": doc_text, "annotations": annotations}, ensure_ascii=False) + "\n")

    print(f"Wrote {n} samples to {out_path} and ground truth to {gt_path}")


if __name__ == "__main__":
    # Example: generate 100 documents in synthetic_data/
    main(out_dir="synthetic_data", n=100, pii_density=0.35)


def _invalid_aadhaar() -> str:
    # generate 12 digits that fail a simple checksum (we'll just alter last digit)
    parts = [_random_digits(4) for _ in range(3)]
    # ensure invalid by setting last group to a fixed invalid pattern
    parts[-1] = "0000"
    return "-".join(parts)


def _invalid_pan() -> str:
    # 10-char alphanumeric that does not follow PAN format (use lower-case / wrong pattern)
    chars = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(10))
    return chars


def _random_10_digit() -> str:
    # random 10-digit number that may not correspond to mobile numbering rules
    return _random_digits(10)


def _company_like_id() -> str:
    # e.g., INV-2025-ABC123 or CIN-like patterns
    return f"INV-{random.randint(2000,2025)}-{_random_digits(4)}"


def generate_negative_samples(out_dir: str = "./synthetic_data/negatives", n: int = 50) -> None:
    """Generate text files that mimic PII formats but are invalid (negative samples).

    The files are useful to test false-positive rates.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    gens = [
        ("AADHAAR_INVALID", _invalid_aadhaar),
        ("PAN_INVALID", _invalid_pan),
        ("RANDOM_10_DIGIT", _random_10_digit),
        ("COMPANY_ID", _company_like_id),
    ]

    for i in range(n):
        ent, gen = random.choice(gens)
        val = gen()
        # craft a sentence that looks plausible but is non-PII
        if ent == "AADHAAR_INVALID":
            text = f"Reference number {val} was used in the invoice."
        elif ent == "PAN_INVALID":
            text = f"Order code {val} processed on {fake.date()} by accounts."
        elif ent == "RANDOM_10_DIGIT":
            text = f"Tracking id {val} scheduled for delivery."
        else:
            text = f"Document ref {val} attached."

        file_name = f"negative_{i:04d}.txt"
        with open(out_path / file_name, "w", encoding="utf-8") as fh:
            fh.write(text)

    print(f"Wrote {n} negative samples to {out_path}")

