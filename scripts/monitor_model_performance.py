"""monitor_model_performance.py

Continuously monitor model performance metrics from Prometheus and detect drift.

Functions:
  - get_live_metrics(prom_url) -> dict: queries Prometheus for precision, recall, f1, latency.
  - check_for_drift(live_metrics, baseline_metrics, thresholds) -> (status, details)
  - trigger_alert(status, details, webhook_url) -> sends webhook POST if provided and logs.

Usage:
  python3 scripts/monitor_model_performance.py --prometheus-url http://prom:9090 --baseline baseline.json --interval 3600 --webhook https://hooks.example.com/notify
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Dict, Optional, Tuple

import requests

DEFAULT_QUERIES = {
    "precision": "avg_over_time(pii_precision[1h])",
    "recall": "avg_over_time(pii_recall[1h])",
    "f1": "avg_over_time(pii_f1[1h])",
    "latency": "histogram_quantile(0.95, sum(rate(pii_processing_duration_seconds_bucket[5m])) by (le))",
}


def query_prometheus(prom_url: str, query: str, timeout: int = 10) -> Optional[float]:
    """Query Prometheus HTTP API /api/v1/query and return a single float value or None."""
    url = prom_url.rstrip("/") + "/api/v1/query"
    try:
        r = requests.get(url, params={"query": query}, timeout=timeout)
        r.raise_for_status()
        payload = r.json()
        if payload.get("status") != "success":
            logging.warning("Prometheus query did not return success: %s", payload)
            return None
        result = payload.get("data", {}).get("result", [])
        if not result:
            return None
        # Use the first result's value (value is [timestamp, value_str])
        val_s = result[0].get("value", [None, None])[1]
        if val_s is None:
            return None
        return float(val_s)
    except Exception as exc:
        logging.debug("Prometheus query failed (%s): %s", query, exc)
        return None


def get_live_metrics(prom_url: str, queries: Dict[str, str] = None) -> Dict[str, Optional[float]]:
    """Fetch latest live metrics from Prometheus. Returns dict with keys: precision, recall, f1, latency."""
    queries = queries or DEFAULT_QUERIES
    out: Dict[str, Optional[float]] = {}
    for k, q in queries.items():
        out[k] = query_prometheus(prom_url, q)
    return out


def _pct_drop(baseline: Optional[float], live: Optional[float]) -> Optional[float]:
    if baseline is None or live is None:
        return None
    if baseline == 0:
        return None
    return (baseline - live) / baseline * 100.0


def _pct_increase(baseline: Optional[float], live: Optional[float]) -> Optional[float]:
    if baseline is None or live is None:
        return None
    if baseline == 0:
        return None
    return (live - baseline) / baseline * 100.0


def check_for_drift(
    live: Dict[str, Optional[float]],
    baseline: Dict[str, float],
    degraded_thresholds: Dict[str, float] = None,
    critical_thresholds: Dict[str, float] = None,
) -> Tuple[str, Dict[str, Optional[float]]]:
    """Compare live metrics against baseline and return status and detail deltas.

    Status rules (defaults):
      - If any metric crosses critical threshold -> 'DRIFT_DETECTED'
      - Else if any metric crosses degraded threshold -> 'DEGRADED'
      - Else -> 'HEALTHY'

    For precision/recall/f1 we compute percent drop (baseline->live). For latency we compute percent increase.
    """
    # default thresholds (percent)
    degraded_thresholds = degraded_thresholds or {"precision": 5.0, "recall": 5.0, "f1": 3.0, "latency": 20.0}
    critical_thresholds = critical_thresholds or {"precision": 10.0, "recall": 10.0, "f1": 7.0, "latency": 50.0}

    details: Dict[str, Optional[float]] = {}
    degraded = False
    critical = False

    # precision, recall, f1: percent drop
    for metric in ("precision", "recall", "f1"):
        pct = _pct_drop(baseline.get(metric), live.get(metric))
        details[f"{metric}_pct_drop"] = pct
        if pct is None:
            continue
        if pct >= critical_thresholds.get(metric, 1000.0):
            critical = True
        elif pct >= degraded_thresholds.get(metric, 1000.0):
            degraded = True

    # latency: percent increase
    lat_inc = _pct_increase(baseline.get("latency"), live.get("latency"))
    details["latency_pct_increase"] = lat_inc
    if lat_inc is not None:
        if lat_inc >= critical_thresholds.get("latency", 1000.0):
            critical = True
        elif lat_inc >= degraded_thresholds.get("latency", 1000.0):
            degraded = True

    if critical:
        return "DRIFT_DETECTED", details
    if degraded:
        return "DEGRADED", details
    return "HEALTHY", details


def trigger_alert(status: str, details: Dict[str, Optional[float]], webhook_url: Optional[str] = None) -> None:
    logging.warning("Model status: %s", status)
    logging.warning("Details: %s", details)
    if webhook_url:
        payload = {"status": status, "details": details}
        try:
            requests.post(webhook_url, json=payload, timeout=10)
            logging.info("Webhook notified: %s", webhook_url)
        except Exception as exc:
            logging.exception("Failed to send webhook: %s", exc)


def load_baseline(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    # Expect numeric baseline values for keys: precision, recall, f1, latency
    return {k: float(data[k]) for k in ("precision", "recall", "f1", "latency")}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monitor model performance and detect drift")
    p.add_argument("--prometheus-url", required=True, help="Prometheus HTTP URL, e.g. http://prometheus:9090")
    p.add_argument("--baseline", required=True, help="Path to JSON file with baseline metrics")
    p.add_argument("--interval", type=int, default=3600, help="Polling interval in seconds (default: 3600)")
    p.add_argument("--webhook", default=None, help="Optional webhook URL to notify when drift is detected")
    p.add_argument("--once", action="store_true", help="Run a single check and exit")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        baseline = load_baseline(args.baseline)
    except Exception as exc:
        logging.exception("Failed to load baseline file: %s", exc)
        return 2

    def run_check() -> None:
        live = get_live_metrics(args.prometheus_url)
        logging.info("Live metrics: %s", live)
        status, details = check_for_drift(live, baseline)
        if status != "HEALTHY":
            trigger_alert(status, details, args.webhook)
        else:
            logging.info("System healthy")

    if args.once:
        run_check()
        return 0

    logging.info("Starting monitoring loop: interval=%s seconds", args.interval)
    try:
        while True:
            run_check()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        logging.info("Interrupted, exiting")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
