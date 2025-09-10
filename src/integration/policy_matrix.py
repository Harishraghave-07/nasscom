"""Policy matrix mapping PII entity names to anonymization policies.

This module centralizes the mapping from detected entity types to the
Presidio anonymizer operator and parameters to use. Each entry in
`POLICY_MATRIX` is a dict with the following keys:

- tier: one of "critical", "high_priority", "contextual", "emerging"
- operator: anonymizer operator name (for example "mask" or "replace")
- params: dict of operator parameters (shape expected by Presidio's
  OperatorConfig-like API; adjust keys if your anonymizer wrapper expects
  different names)

Why a central policy matrix?
- Keeps anonymization behavior auditable and reviewable by compliance teams.
- Allows per-entity tuning (for example, use replace for small labels and
  masking for long identifiers).
- Makes it easy to programmatically generate Presidio AnonymizerEngine
  OperatorConfig objects when constructing anonymization pipelines.

Examples
--------
>>> from src.integration.policy_matrix import POLICY_MATRIX
>>> POLICY_MATRIX["CREDIT_CARD"]["operator"]
'mask'
>>> POLICY_MATRIX["AADHAAR_NUMBER"]["tier"]
'critical'

Notes on params
---------------
This file provides conservative defaults. Operators and parameter keys are
chosen to be compatible with common Presidio anonymizer operator configs
(for example: mask -> {"masking_char": "X", "chars_to_mask": 6, "from_end": True},
 replace -> {"new_value": "[REDACTED]"}). If your anonymizer wrapper
expects different param names, adapt the translation layer that builds the
OperatorConfig objects.
"""

from __future__ import annotations

from typing import Dict, Any


POLICY_MATRIX: Dict[str, Dict[str, Any]] = {
    # Critical identifiers (very high risk — strict handling)
    "AADHAAR_NUMBER": {
        "tier": "critical",
        "operator": "mask",
        "params": {"masking_char": "X", "chars_to_mask": 8, "from_end": True},
    },
    "PAN_NUMBER": {
        "tier": "critical",
        "operator": "mask",
        "params": {"masking_char": "X", "chars_to_mask": 5, "from_end": True},
    },
    "GSTIN": {
        "tier": "critical",
        "operator": "mask",
        "params": {"masking_char": "X", "chars_to_mask": 6, "from_end": True},
    },
    "MRN_MEDICAL_RECORD_NUMBER": {
        "tier": "critical",
        "operator": "replace",
        "params": {"new_value": "[REDACTED_MRN]"},
    },

    # High-priority financial and account identifiers
    "CREDIT_CARD": {
        "tier": "high_priority",
        "operator": "mask",
        "params": {"masking_char": "*", "chars_to_mask": 12, "from_end": True},
    },
    "BANK_ACCOUNT": {
        "tier": "high_priority",
        "operator": "mask",
        "params": {"masking_char": "*", "chars_to_mask": 6, "from_end": True},
    },
    "PHONE_NUMBER": {
        "tier": "high_priority",
        "operator": "mask",
        "params": {"masking_char": "*", "chars_to_mask": 6, "from_end": True},
    },
    "EMAIL_ADDRESS": {
        "tier": "high_priority",
        "operator": "replace",
        "params": {"new_value": "[REDACTED_EMAIL]"},
    },

    # Contextual fields (may require contextual checking or coalescing spans)
    "NAME": {
        "tier": "contextual",
        "operator": "replace",
        "params": {"new_value": "[REDACTED_NAME]"},
    },
    "DATE_OF_BIRTH": {
        "tier": "contextual",
        "operator": "replace",
        "params": {"new_value": "[REDACTED_DOB]"},
    },
    "ADDRESS": {
        "tier": "contextual",
        "operator": "replace",
        "params": {"new_value": "[REDACTED_ADDRESS]"},
    },
    "IP_ADDRESS": {
        "tier": "contextual",
        "operator": "mask",
        "params": {"masking_char": "0", "chars_to_mask": 6, "from_end": True},
    },
    "URL": {
        "tier": "contextual",
        "operator": "replace",
        "params": {"new_value": "[REDACTED_URL]"},
    },

    # Emerging or privacy-adjacent fields (lower confidence analyzers)
    "BIOMETRIC": {
        "tier": "emerging",
        "operator": "replace",
        "params": {"new_value": "[REDACTED_BIOMETRIC]"},
    },
    "FULL_FACE_PHOTO": {
        "tier": "emerging",
        "operator": "replace",
        "params": {"new_value": "[REDACTED_FACE]"},
    },
    "DEVICE_ID": {
        "tier": "emerging",
        "operator": "mask",
        "params": {"masking_char": "X", "chars_to_mask": 8, "from_end": False},
    },

    # Other commonly-detected identifiers
    "VOTER_ID": {
        "tier": "high_priority",
        "operator": "mask",
        "params": {"masking_char": "X", "chars_to_mask": 6, "from_end": True},
    },
    "PASSPORT_NUMBER": {
        "tier": "high_priority",
        "operator": "mask",
        "params": {"masking_char": "X", "chars_to_mask": 6, "from_end": True},
    },
    "SOCIAL_SECURITY_NUMBER": {
        "tier": "critical",
        "operator": "mask",
        "params": {"masking_char": "X", "chars_to_mask": 5, "from_end": True},
    },
    "LICENSE_NUMBER": {
        "tier": "contextual",
        "operator": "replace",
        "params": {"new_value": "[REDACTED_LICENSE]"},
    },

    # Misc / fallback
    "OTHER_UNIQUE_IDENTIFIER": {
        "tier": "contextual",
        "operator": "replace",
        "params": {"new_value": "[REDACTED_ID]"},
    },
}


__all__ = ["POLICY_MATRIX"]
