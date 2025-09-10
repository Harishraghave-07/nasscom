"""Config loader utilities to merge operator overrides into POLICY_MATRIX.

This module reads an optional `operator_overrides` block from the provided
configuration and merges those overrides into the runtime `POLICY_MATRIX`.

Example override shape (in YAML or dict form):

operator_overrides:
  CREDIT_CARD:
    operator: mask
    params:
      masking_char: "#"
      chars_to_mask: 12

Functions
---------
- apply_operator_overrides(config, policy_matrix): mutate `policy_matrix` in-place
- get_operator(entity_type, policy_matrix): return the final operator dict for entity

Doc examples
------------
>>> from src.integration.config_loader import apply_operator_overrides, get_operator
>>> from src.integration.policy_matrix import POLICY_MATRIX
>>> cfg = {"operator_overrides": {"CREDIT_CARD": {"operator": "mask", "params": {"masking_char": "#", "chars_to_mask": 12}}}}
>>> apply_operator_overrides(cfg, POLICY_MATRIX)
>>> get_operator("CREDIT_CARD", POLICY_MATRIX)["operator"]
'mask'
"""

from __future__ import annotations

from typing import Dict, Any


def _normalize_override(o: Any) -> Dict[str, Any]:
    # Accept either {'operator': 'mask', 'masking_char': '#', ...}
    # or nested under 'params'. Normalize to {'operator':..., 'params': {...}}
    if not isinstance(o, dict):
        return {}
    if "params" in o:
        return {"operator": o.get("operator"), "params": o.get("params", {})}
    # flatten keys that look like params
    params = {k: v for k, v in o.items() if k != "operator"}
    return {"operator": o.get("operator"), "params": params}


def apply_operator_overrides(config: Dict[str, Any], policy_matrix: Dict[str, Dict[str, Any]]) -> None:
    """Apply operator_overrides found in `config` to `policy_matrix` in-place.

    - config: dict-like object possibly containing 'operator_overrides'
    - policy_matrix: the POLICY_MATRIX dict to mutate

    The function is tolerant to missing keys and will only update entries
    that exist in the policy_matrix or add new ones.
    """
    if not config:
        return
    overrides = config.get("operator_overrides") or {}
    if not isinstance(overrides, dict):
        return
    for ent, override in overrides.items():
        key = ent.upper()
        norm = _normalize_override(override)
        if not norm:
            continue
        existing = policy_matrix.get(key) or policy_matrix.get(ent)
        if existing:
            # merge operator and params
            if norm.get("operator"):
                existing["operator"] = norm["operator"]
            if isinstance(norm.get("params"), dict):
                existing_params = existing.get("params", {}) or {}
                existing_params.update(norm.get("params", {}))
                existing["params"] = existing_params
        else:
            # add new entry
            policy_matrix[key] = {"tier": "contextual", "operator": norm.get("operator", "replace"), "params": norm.get("params", {})}


def get_operator(entity_type: str, policy_matrix: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Return the final operator config for `entity_type` from `policy_matrix`.

    Returns a dict with keys: 'operator' and 'params'. If the entity is not
    found, returns a default replace operator.
    """
    if not entity_type:
        return {"operator": "replace", "params": {"new_value": "[REDACTED]"}}
    key = entity_type.upper()
    entry = policy_matrix.get(key) or policy_matrix.get(entity_type)
    if not entry:
        return {"operator": "replace", "params": {"new_value": "[REDACTED]"}}
    return {"operator": entry.get("operator", "replace"), "params": entry.get("params", {}) or {}}


__all__ = ["apply_operator_overrides", "get_operator"]
