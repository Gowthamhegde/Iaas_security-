"""
Stage 4 — Inference Service  (Algorithm 1 implementation)
Runs the ML-based flow classification pipeline:
  1. Capture traffic (live Scapy flow aggregation or synthetic replay)
  2. Extract NSL-KDD-compatible flow features
  3. Preprocess with train-time encoders/scaler
  4. Classify with Random Forest
  5. Emit alerts when y^ != Normal and prob >= tau

Live-server mode:
  python -m inference.service --live --interface eth0
  python pipeline.py --infer --live --interface eth0
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.preprocess import (
    load_preprocessor,
    preprocess_single_flow,
)
from model.train import (
    load_model,
    load_metrics,
    validate_artifacts,
    ARTIFACTS_DIR,
)
from inference.flow_extractor import LiveFlowExtractor

logger = logging.getLogger(__name__)

DEFAULT_TAU = 0.60
ALERTS_DIR = Path(__file__).parent.parent / "logs"
ALERTS_PATH = ALERTS_DIR / "alerts.jsonl"

_event_queue: queue.Queue = queue.Queue(maxsize=2000)


def get_event_queue():
    return _event_queue


# ── Synthetic flow generator (demo / offline) ─────────────────────────────────

_PROTOCOLS = ["tcp", "udp", "icmp"]
_SERVICES = ["http", "ftp", "smtp", "ssh", "dns", "private", "other"]
_FLAGS = ["SF", "S0", "REJ", "RSTO", "SH", "S1", "S2", "S3", "OTH"]

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
    if true_label is None:
        true_label = random.choices(
            ["Normal", "DoS", "Probe", "R2L", "U2R"],
            weights=[0.53, 0.37, 0.09, 0.009, 0.001],
            k=1,
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
    service = random.choice(_SERVICES)
    flag = _weighted_choice(t["flag_bias"])

    return {
        "duration": duration,
        "protocol_type": protocol,
        "service": service,
        "flag": flag,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "land": 0,
        "wrong_fragment": 0,
        "urgent": 0,
        "hot": random.randint(0, 3),
        "num_failed_logins": 0,
        "logged_in": 1 if true_label == "Normal" else 0,
        "num_compromised": 0,
        "root_shell": 1 if true_label == "U2R" else 0,
        "su_attempted": 0,
        "num_root": 0,
        "num_file_creations": 0,
        "num_shells": 0,
        "num_access_files": 0,
        "num_outbound_cmds": 0,
        "is_host_login": 0,
        "is_guest_login": 0,
        "count": count,
        "srv_count": srv_count,
        "serror_rate": t["serror_rate"] + random.uniform(-0.02, 0.02),
        "srv_serror_rate": t["serror_rate"] + random.uniform(-0.02, 0.02),
        "rerror_rate": random.uniform(0, 0.1),
        "srv_rerror_rate": random.uniform(0, 0.1),
        "same_srv_rate": min(1.0, max(0.0, t["same_srv_rate"] + random.uniform(-0.1, 0.1))),
        "diff_srv_rate": random.uniform(0, 0.3),
        "srv_diff_host_rate": random.uniform(0, 0.3),
        "dst_host_count": random.randint(1, 255),
        "dst_host_srv_count": random.randint(1, 255),
        "dst_host_same_srv_rate": random.uniform(0.5, 1.0),
        "dst_host_diff_srv_rate": random.uniform(0, 0.3),
        "dst_host_same_src_port_rate": random.uniform(0, 1.0),
        "dst_host_srv_diff_host_rate": random.uniform(0, 0.2),
        "dst_host_serror_rate": t["serror_rate"],
        "dst_host_srv_serror_rate": t["serror_rate"],
        "dst_host_rerror_rate": random.uniform(0, 0.1),
        "dst_host_srv_rerror_rate": random.uniform(0, 0.1),
        "_true_label": true_label,
        "_timestamp": datetime.now().isoformat(),
        "_src_ip": f"10.0.{random.randint(0,255)}.{random.randint(1,254)}",
        "_dst_ip": f"192.168.1.{random.randint(1,254)}",
        "_src_port": random.randint(1024, 65535),
        "_dst_port": random.choice([80, 443, 22, 21, 25, 53, 8080]),
        "_capture_mode": "simulate",
    }


def _append_alert(event, path=ALERTS_PATH):
    """Append alert events to a JSONL sink for SIEM / log shipping."""
    if not event.get("is_alert"):
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=str) + "\n")


class InferenceService:
    """
    Algorithm 1: ML-Based Flow Classification for live servers or simulation.

    Args:
        simulate:     True = synthetic flows; False = live NIC capture
        interface:    network interface for live capture (None = default)
        tau:          confidence threshold for alerting
        rate:         simulated flows per second (simulation only)
        flow_timeout: idle seconds before a live flow is scored
        alert_log:    path for JSONL alert sink (None disables)
    """

    def __init__(
        self,
        simulate=True,
        interface=None,
        tau=DEFAULT_TAU,
        rate=2.0,
        flow_timeout=30.0,
        alert_log=ALERTS_PATH,
    ):
        self.simulate = simulate
        self.interface = interface
        self.tau = tau
        self.rate = rate
        self.flow_timeout = flow_timeout
        self.alert_log = Path(alert_log) if alert_log else None
        self._running = False
        self._thread = None
        self._extractor = None

        logger.info("[Inference] Loading train-time preprocessor ...")
        prep = load_preprocessor()
        self._encoders = prep["encoders"]
        self._scaler = prep["scaler"]
        self._feature_names = prep["feature_names"]

        logger.info("[Inference] Loading trained model ...")
        self._model = load_model()
        validate_artifacts(self._model, load_metrics(), self._feature_names)
        self._classes = list(self._model.classes_)
        mode = "simulate" if simulate else f"live({interface or 'default'})"
        logger.info(f"[Inference] Classes: {self._classes}")
        logger.info(f"[Inference] Mode: {mode} | tau={self.tau}")

    def _classify_flow(self, raw_flow):
        X = preprocess_single_flow(
            {k: v for k, v in raw_flow.items() if not k.startswith("_")},
            self._encoders,
            self._scaler,
            self._feature_names,
        )

        y_hat = self._model.predict(X)[0]
        proba = self._model.predict_proba(X)[0]
        prob = float(proba[self._classes.index(y_hat)])

        importances = dict(
            sorted(
                zip(self._feature_names, self._model.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        is_alert = (y_hat != "Normal" and prob >= self.tau)

        event = {
            "timestamp": raw_flow.get("_timestamp", datetime.now().isoformat()),
            "src_ip": raw_flow.get("_src_ip", "0.0.0.0"),
            "dst_ip": raw_flow.get("_dst_ip", "0.0.0.0"),
            "src_port": raw_flow.get("_src_port", 0),
            "dst_port": raw_flow.get("_dst_port", 0),
            "protocol": raw_flow.get("protocol_type", "tcp"),
            "service": raw_flow.get("service", "http"),
            "predicted": y_hat,
            "probability": round(prob, 4),
            "probabilities": {
                cls: round(float(p), 4)
                for cls, p in zip(self._classes, proba)
            },
            "is_alert": is_alert,
            "true_label": raw_flow.get("_true_label", "Unknown"),
            "top_features": dict(list(importances.items())[:10]),
            "capture_mode": raw_flow.get("_capture_mode", "unknown"),
            "raw_features": {
                k: v for k, v in raw_flow.items() if not k.startswith("_")
            },
        }

        if is_alert:
            logger.warning(
                f"[ALERT] {y_hat} from {event['src_ip']} -> {event['dst_ip']} "
                f"(prob={prob:.2f})"
            )
            if self.alert_log:
                _append_alert(event, self.alert_log)

        return event

    def _enqueue(self, event):
        try:
            _event_queue.put_nowait(event)
        except queue.Full:
            try:
                _event_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                _event_queue.put_nowait(event)
            except queue.Full:
                pass

    def _on_live_flow(self, raw_flow):
        if not self._running:
            return
        event = self._classify_flow(raw_flow)
        self._enqueue(event)

    def _simulation_loop(self):
        interval = 1.0 / max(self.rate, 0.1)
        while self._running:
            raw_flow = _generate_synthetic_flow()
            event = self._classify_flow(raw_flow)
            self._enqueue(event)
            time.sleep(interval)

    def start(self):
        if self._running:
            logger.warning("[Inference] Already running.")
            return
        self._running = True

        if self.simulate:
            self._thread = threading.Thread(
                target=self._simulation_loop, daemon=True, name="inference-sim"
            )
            self._thread.start()
            logger.info("[Inference] Service started (simulation mode).")
            return

        ifaces = LiveFlowExtractor.list_interfaces()
        if ifaces:
            logger.info(f"[Inference] Available interfaces: {ifaces}")

        self._extractor = LiveFlowExtractor(
            interface=self.interface,
            flow_timeout=self.flow_timeout,
            on_flow=self._on_live_flow,
        )
        self._extractor.start()
        logger.info(
            f"[Inference] Service started (LIVE capture on "
            f"{self.interface or 'default'}). Alerts -> {self.alert_log}"
        )

    def stop(self):
        self._running = False
        if self._extractor:
            self._extractor.stop()
            self._extractor = None
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("[Inference] Service stopped.")

    def classify_once(self, raw_flow):
        return self._classify_flow(raw_flow)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="IaaS Security Inference Service")
    parser.add_argument(
        "--live", action="store_true",
        help="Capture live traffic from a network interface (server mode)",
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Force synthetic flow mode",
    )
    parser.add_argument("--interface", default=None, help="NIC for live capture")
    parser.add_argument("--list-ifaces", action="store_true", help="List NICs and exit")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--rate", type=float, default=2.0)
    parser.add_argument("--flow-timeout", type=float, default=30.0)
    parser.add_argument("--duration", type=int, default=0, help="0 = run until Ctrl-C")
    parser.add_argument("--alert-log", default=str(ALERTS_PATH))
    args = parser.parse_args()

    if args.list_ifaces:
        ifaces = LiveFlowExtractor.list_interfaces()
        print("Available interfaces:" if ifaces else "No interfaces found (is Scapy/Npcap installed?)")
        for name in ifaces:
            print(f"  - {name}")
        sys.exit(0)

    use_simulate = args.simulate or not args.live
    if args.live and args.simulate:
        logger.warning("--live and --simulate both set; using --live")
        use_simulate = False

    svc = InferenceService(
        simulate=use_simulate,
        interface=args.interface,
        tau=args.tau,
        rate=args.rate,
        flow_timeout=args.flow_timeout,
        alert_log=args.alert_log,
    )
    svc.start()

    q = get_event_queue()
    end = time.time() + args.duration if args.duration else None
    try:
        while True:
            if end and time.time() >= end:
                break
            try:
                evt = q.get(timeout=1.0)
                tag = "[ALERT]" if evt["is_alert"] else "      "
                print(
                    f"{tag} {evt['timestamp'][:19]} | "
                    f"{evt['predicted']:8s} ({evt['probability']:.2f}) | "
                    f"{evt['src_ip']}:{evt['src_port']} -> "
                    f"{evt['dst_ip']}:{evt['dst_port']} | {evt.get('capture_mode')}"
                )
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        svc.stop()
