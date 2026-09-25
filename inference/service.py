"""
Stage 4 — Inference Service  (Algorithm 1 implementation)
Runs the ML-based flow classification pipeline:
  1. Capture traffic (Scapy or simulated pcap/flow replay)
  2. Extract flow features (CICFlowMeter-style simulation on Windows)
  3. Preprocess features
  4. Classify with Random Forest
  5. Emit alerts when y^ != Normal and prob >= tau

On Windows, live Scapy capture requires WinPcap/Npcap and admin privileges.
The service gracefully falls back to synthetic flow simulation when capture
is unavailable or --simulate flag is passed.
"""

import sys
import time
import random
import logging
import threading
import queue
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.preprocess import (
    load_and_preprocess,
    preprocess_single_flow,
    CATEGORICAL_COLS,
)
from model.train import load_model, load_metrics, ARTIFACTS_DIR

logger = logging.getLogger(__name__)

# Classification confidence threshold (tau in Algorithm 1)
DEFAULT_TAU = 0.60

# ── Shared alert/result queue (Dashboard reads from this) ─────────────────────
_event_queue: queue.Queue = queue.Queue(maxsize=2000)


def get_event_queue():
    return _event_queue


# ── Synthetic flow generator ──────────────────────────────────────────────────
# Generates realistic-looking flows by sampling from NSL-KDD test statistics.

_PROTOCOLS = ["tcp", "udp", "icmp"]
_SERVICES   = ["http", "ftp", "smtp", "ssh", "dns", "private", "other"]
_FLAGS      = ["SF", "S0", "REJ", "RSTO", "SH", "S1", "S2", "S3", "OTH"]

# Per-class flow templates  (approximate NSL-KDD feature distributions)
_TEMPLATES = {
    "Normal": {
        "duration": (10, 200), "src_bytes": (500, 5000), "dst_bytes": (200, 3000),
        "count": (1, 50), "srv_count": (1, 40), "serror_rate": 0.0,
        "same_srv_rate": 0.9, "flag_bias": {"SF": 0.85, "S1": 0.10, "OTH": 0.05},
    },
    "DoS": {
        "duration": (0, 2), "src_bytes": (0, 200), "dst_bytes": (0, 0),
        "count": (200, 511), "srv_count": (200, 511), "serror_rate": 0.98,
        "same_srv_rate": 0.99, "flag_bias": {"S0": 0.90, "REJ": 0.09, "OTH": 0.01},
    },
    "Probe": {
        "duration": (0, 5), "src_bytes": (28, 400), "dst_bytes": (0, 100),
        "count": (50, 200), "srv_count": (1, 30), "serror_rate": 0.10,
        "same_srv_rate": 0.10, "flag_bias": {"S0": 0.50, "REJ": 0.30, "SF": 0.20},
    },
    "R2L": {
        "duration": (30, 300), "src_bytes": (1000, 30000), "dst_bytes": (100, 2000),
        "count": (1, 5), "srv_count": (1, 5), "serror_rate": 0.0,
        "same_srv_rate": 0.5, "flag_bias": {"SF": 0.60, "RSTO": 0.30, "OTH": 0.10},
    },
    "U2R": {
        "duration": (5, 60), "src_bytes": (200, 5000), "dst_bytes": (100, 1000),
        "count": (1, 3), "srv_count": (1, 3), "serror_rate": 0.0,
        "same_srv_rate": 0.5, "flag_bias": {"SF": 0.75, "RSTO": 0.20, "OTH": 0.05},
    },
}


def _weighted_choice(bias_dict):
    keys = list(bias_dict.keys())
    weights = list(bias_dict.values())
    return random.choices(keys, weights=weights, k=1)[0]


def _generate_synthetic_flow(true_label=None):
    """
    Generate a single synthetic network flow record matching NSL-KDD schema.
    If true_label is None, randomly pick a class with realistic class priors.
    """
    # Class priors from Table III
    if true_label is None:
        true_label = random.choices(
            ["Normal", "DoS", "Probe", "R2L", "U2R"],
            weights=[0.53, 0.37, 0.09, 0.009, 0.001],
            k=1
        )[0]

    t = _TEMPLATES[true_label]
    lo, hi = t["duration"]
    duration = random.randint(lo, hi)
    lo, hi = t["src_bytes"]
    src_bytes = random.randint(lo, hi)
    lo, hi = t["dst_bytes"]
    dst_bytes = random.randint(lo, hi)
    lo, hi = t["count"]
    count = random.randint(lo, hi)
    lo, hi = t["srv_count"]
    srv_count = random.randint(lo, hi)

    protocol = random.choice(_PROTOCOLS)
    service  = random.choice(_SERVICES)
    flag     = _weighted_choice(t["flag_bias"])

    flow = {
        "duration":              duration,
        "protocol_type":         protocol,
        "service":               service,
        "flag":                  flag,
        "src_bytes":             src_bytes,
        "dst_bytes":             dst_bytes,
        "land":                  0,
        "wrong_fragment":        0,
        "urgent":                0,
        "hot":                   random.randint(0, 3),
        "num_failed_logins":     0,
        "logged_in":             1 if true_label == "Normal" else 0,
        "num_compromised":       0,
        "root_shell":            1 if true_label == "U2R" else 0,
        "su_attempted":          0,
        "num_root":              0,
        "num_file_creations":    0,
        "num_shells":            0,
        "num_access_files":      0,
        "num_outbound_cmds":     0,
        "is_host_login":         0,
        "is_guest_login":        0,
        "count":                 count,
        "srv_count":             srv_count,
        "serror_rate":           t["serror_rate"] + random.uniform(-0.02, 0.02),
        "srv_serror_rate":       t["serror_rate"] + random.uniform(-0.02, 0.02),
        "rerror_rate":           random.uniform(0, 0.1),
        "srv_rerror_rate":       random.uniform(0, 0.1),
        "same_srv_rate":         min(1.0, max(0.0, t["same_srv_rate"] + random.uniform(-0.1, 0.1))),
        "diff_srv_rate":         random.uniform(0, 0.3),
        "srv_diff_host_rate":    random.uniform(0, 0.3),
        "dst_host_count":        random.randint(1, 255),
        "dst_host_srv_count":    random.randint(1, 255),
        "dst_host_same_srv_rate": random.uniform(0.5, 1.0),
        "dst_host_diff_srv_rate": random.uniform(0, 0.3),
        "dst_host_same_src_port_rate": random.uniform(0, 1.0),
        "dst_host_srv_diff_host_rate": random.uniform(0, 0.2),
        "dst_host_serror_rate":  t["serror_rate"],
        "dst_host_srv_serror_rate": t["serror_rate"],
        "dst_host_rerror_rate":  random.uniform(0, 0.1),
        "dst_host_srv_rerror_rate": random.uniform(0, 0.1),
        # metadata (not model features)
        "_true_label":  true_label,
        "_timestamp":   datetime.now().isoformat(),
        "_src_ip":      f"10.0.{random.randint(0,255)}.{random.randint(1,254)}",
        "_dst_ip":      f"192.168.1.{random.randint(1,254)}",
        "_src_port":    random.randint(1024, 65535),
        "_dst_port":    random.choice([80, 443, 22, 21, 25, 53, 8080]),
    }
    return flow


# ── Scapy live capture (optional, requires Npcap + admin) ─────────────────────

def _try_live_capture(interface, packet_count=100):
    """Attempt live Scapy capture; returns list of raw flow dicts or None."""
    try:
        from scapy.all import sniff, IP, TCP, UDP
        logger.info(f"[Capture] Starting Scapy capture on {interface} ...")
        packets = sniff(iface=interface, count=packet_count, timeout=10)
        flows = []
        for pkt in packets:
            if IP in pkt:
                flow = {
                    "duration":          0,
                    "protocol_type":     "tcp" if TCP in pkt else "udp" if UDP in pkt else "icmp",
                    "service":           "http",
                    "flag":              "SF",
                    "src_bytes":         len(pkt),
                    "dst_bytes":         0,
                    "_src_ip":           pkt[IP].src,
                    "_dst_ip":           pkt[IP].dst,
                    "_src_port":         pkt[TCP].sport if TCP in pkt else 0,
                    "_dst_port":         pkt[TCP].dport if TCP in pkt else 0,
                    "_timestamp":        datetime.now().isoformat(),
                    "_true_label":       "Unknown",
                }
                # Fill remaining NSL-KDD fields with 0
                flows.append(flow)
        return flows
    except Exception as e:
        logger.warning(f"[Capture] Live capture failed: {e}. Falling back to simulation.")
        return None


# ── Core inference pipeline (Algorithm 1) ─────────────────────────────────────

class InferenceService:
    """
    Implements Algorithm 1: ML-Based Flow Classification Pipeline.

    Args:
        simulate:     use synthetic flows instead of live capture
        interface:    network interface for live capture
        tau:          confidence threshold for alerting
        rate:         simulated flows per second (simulation mode only)
    """

    def __init__(self, simulate=True, interface=None, tau=DEFAULT_TAU, rate=2.0):
        self.simulate  = simulate
        self.interface = interface
        self.tau       = tau
        self.rate      = rate
        self._running  = False
        self._thread   = None

        # Load preprocessing artifacts
        logger.info("[Inference] Loading preprocessing artifacts ...")
        self._data = load_and_preprocess()
        self._encoders      = self._data["encoders"]
        self._scaler        = self._data["scaler"]
        self._feature_names = self._data["feature_names"]

        # Load trained model (Algorithm 1, line 4)
        logger.info("[Inference] Loading trained model ...")
        self._model = load_model()
        self._classes = list(self._model.classes_)
        logger.info(f"[Inference] Classes: {self._classes}")
        logger.info(f"[Inference] Tau threshold: {self.tau}")

    def _classify_flow(self, raw_flow):
        """
        Algorithm 1, lines 5-10:
        Preprocess -> predict -> threshold -> alert.
        """
        X = preprocess_single_flow(
            {k: v for k, v in raw_flow.items() if not k.startswith("_")},
            self._encoders,
            self._scaler,
            self._feature_names,
        )

        # line 6
        y_hat = self._model.predict(X)[0]
        proba  = self._model.predict_proba(X)[0]
        prob   = float(proba[self._classes.index(y_hat)])

        # line 8: feature importances
        importances = dict(
            sorted(
                zip(self._feature_names, self._model.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        is_alert = (y_hat != "Normal" and prob >= self.tau)

        event = {
            "timestamp":    raw_flow.get("_timestamp", datetime.now().isoformat()),
            "src_ip":       raw_flow.get("_src_ip", "0.0.0.0"),
            "dst_ip":       raw_flow.get("_dst_ip", "0.0.0.0"),
            "src_port":     raw_flow.get("_src_port", 0),
            "dst_port":     raw_flow.get("_dst_port", 0),
            "protocol":     raw_flow.get("protocol_type", "tcp"),
            "service":      raw_flow.get("service", "http"),
            "predicted":    y_hat,
            "probability":  round(prob, 4),
            "probabilities": {
                cls: round(float(p), 4)
                for cls, p in zip(self._classes, proba)
            },
            "is_alert":     is_alert,
            "true_label":   raw_flow.get("_true_label", "Unknown"),
            "top_features": dict(list(importances.items())[:10]),
            "raw_features": {
                k: v for k, v in raw_flow.items() if not k.startswith("_")
            },
        }

        if is_alert:
            logger.warning(
                f"[ALERT] {y_hat} from {event['src_ip']} -> {event['dst_ip']} "
                f"(prob={prob:.2f})"
            )

        return event

    def _simulation_loop(self):
        """Generate synthetic flows at self.rate flows/sec."""
        interval = 1.0 / max(self.rate, 0.1)
        while self._running:
            raw_flow = _generate_synthetic_flow()
            event = self._classify_flow(raw_flow)
            try:
                _event_queue.put_nowait(event)
            except queue.Full:
                _event_queue.get_nowait()   # drop oldest
                _event_queue.put_nowait(event)
            time.sleep(interval)

    def _capture_loop(self):
        """Live capture loop (batch-by-batch)."""
        while self._running:
            flows = _try_live_capture(self.interface, packet_count=50)
            if flows is None:
                # Fallback to simulation for this batch
                flows = [_generate_synthetic_flow() for _ in range(10)]
            for raw_flow in flows:
                if not self._running:
                    break
                event = self._classify_flow(raw_flow)
                try:
                    _event_queue.put_nowait(event)
                except queue.Full:
                    _event_queue.get_nowait()
                    _event_queue.put_nowait(event)
            time.sleep(1.0)

    def start(self):
        """Start the inference pipeline in a background thread."""
        if self._running:
            logger.warning("[Inference] Already running.")
            return
        self._running = True
        target = self._simulation_loop if self.simulate else self._capture_loop
        self._thread = threading.Thread(target=target, daemon=True, name="inference")
        self._thread.start()
        logger.info(f"[Inference] Service started ({'simulation' if self.simulate else 'live capture'} mode).")

    def stop(self):
        """Stop the inference pipeline."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("[Inference] Service stopped.")

    def classify_once(self, raw_flow):
        """Synchronously classify a single flow dict (for testing)."""
        return self._classify_flow(raw_flow)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="IaaS Security Inference Service")
    parser.add_argument("--simulate", action="store_true", default=True,
                        help="Use synthetic flows (default on Windows)")
    parser.add_argument("--interface", default=None, help="Network interface for live capture")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU, help="Alert confidence threshold")
    parser.add_argument("--rate", type=float, default=2.0, help="Flows per second (simulation mode)")
    parser.add_argument("--duration", type=int, default=30, help="Run for N seconds then exit")
    args = parser.parse_args()

    svc = InferenceService(
        simulate=args.simulate,
        interface=args.interface,
        tau=args.tau,
        rate=args.rate,
    )
    svc.start()

    q = get_event_queue()
    end = time.time() + args.duration
    while time.time() < end:
        try:
            evt = q.get(timeout=1.0)
            tag = "[ALERT]" if evt["is_alert"] else "      "
            print(f"{tag} {evt['timestamp']} | {evt['predicted']:8s} ({evt['probability']:.2f}) | "
                  f"{evt['src_ip']} -> {evt['dst_ip']}")
        except queue.Empty:
            pass

    svc.stop()
