"""
pipeline.py — Top-level orchestrator for the ML-Based IaaS IDS system.
Implements the full Algorithm 1 lifecycle:
  Stage 1: Environment check
  Stage 2: Data acquisition & preprocessing
  Stage 3: Model training (with optional grid search)
  Stage 4: Inference service start
  Stage 5: Dashboard launch

Usage:
  python pipeline.py --train          # Stages 1-3 only (train model)
  python pipeline.py --infer          # Stages 4 only (run inference, no dashboard)
  python pipeline.py --dashboard      # Stage 5 (launch dashboard)
  python pipeline.py                  # Full pipeline: train -> infer + dashboard
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def stage1_check():
    """Stage 1: Verify environment and dependencies."""
    logger.info("=" * 60)
    logger.info("STAGE 1 — Environment Check")
    logger.info("=" * 60)
    # Map display name -> actual importable module name
    packages = {
        "scikit-learn": "sklearn",
        "pandas":       "pandas",
        "numpy":        "numpy",
        "joblib":       "joblib",
        "streamlit":    "streamlit",
        "plotly":       "plotly",
        "requests":     "requests",
        "tqdm":         "tqdm",
    }
    missing = []
    for display, module in packages.items():
        try:
            __import__(module)
            logger.info(f"  [OK] {display}")
        except ImportError:
            logger.error(f"  [MISSING] {display}")
            missing.append(display)
    if missing:
        logger.error(f"\nMissing packages: {missing}")
        logger.error("Run:  pip install -r requirements.txt")
        sys.exit(1)
    logger.info("Environment OK.\n")


def stage2_preprocess():
    """Stage 2: Download NSL-KDD and run preprocessing pipeline."""
    logger.info("=" * 60)
    logger.info("STAGE 2 — Data Acquisition & Preprocessing")
    logger.info("=" * 60)
    from data.preprocess import load_and_preprocess
    data = load_and_preprocess()
    logger.info(f"Train samples: {data['X_train'].shape[0]:,}")
    logger.info(f"Test  samples: {data['X_test'].shape[0]:,}")
    logger.info(f"Features:      {data['X_train'].shape[1]}")
    logger.info("Preprocessing complete.\n")
    return data


def stage3_train(data=None, quick=False):
    """Stage 3: Train Random Forest with grid search + 5-fold CV."""
    logger.info("=" * 60)
    logger.info("STAGE 3 — Model Development")
    logger.info("=" * 60)
    if data is None:
        data = stage2_preprocess()
    from model.train import train
    model, metrics = train(
        data["X_train"], data["y_train"],
        data["X_test"],  data["y_test"],
        feature_names=data["feature_names"],
        quick=quick,
    )
    logger.info(f"Test Accuracy : {metrics['accuracy']:.4f}")
    logger.info(f"F1 (macro)    : {metrics['f1_macro']:.4f}")
    logger.info("Model serialized to model/artifacts/random_forest.pkl\n")
    return model, metrics


def stage4_infer(tau=0.60, rate=3.0, duration=None):
    """Stage 4: Start the inference service (blocks until duration expires or Ctrl-C)."""
    logger.info("=" * 60)
    logger.info("STAGE 4 — Inference Service")
    logger.info("=" * 60)
    from inference.service import InferenceService, get_event_queue
    import queue as q

    svc = InferenceService(simulate=True, tau=tau, rate=rate)
    svc.start()

    event_queue = get_event_queue()
    start = time.time()
    try:
        while True:
            if duration and (time.time() - start) > duration:
                break
            try:
                evt = event_queue.get(timeout=1.0)
                tag = "[ALERT]" if evt["is_alert"] else "      "
                print(
                    f"{tag} {evt['timestamp'][:19]} | "
                    f"{evt['predicted']:8s} ({evt['probability']:.2f}) | "
                    f"{evt['src_ip']} -> {evt['dst_ip']}"
                )
            except q.Empty:
                pass
    except KeyboardInterrupt:
        logger.info("\nStopped by user.")
    finally:
        svc.stop()


def stage5_dashboard():
    """Stage 5: Launch the Streamlit dashboard."""
    logger.info("=" * 60)
    logger.info("STAGE 5 — Dashboard Deployment")
    logger.info("=" * 60)
    app_path = ROOT / "dashboard" / "app.py"
    logger.info(f"Launching Streamlit at http://localhost:8501")
    logger.info("Press Ctrl+C to stop.\n")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", "8501",
        "--server.headless", "false",
        "--theme.base", "dark",
    ])


def main():
    parser = argparse.ArgumentParser(description="IaaS ML-IDS Pipeline")
    parser.add_argument("--train",     action="store_true", help="Run Stages 1-3 (training)")
    parser.add_argument("--infer",     action="store_true", help="Run Stage 4 (inference only)")
    parser.add_argument("--dashboard", action="store_true", help="Run Stage 5 (dashboard only)")
    parser.add_argument("--quick",     action="store_true", help="Fast training (single RF config, no grid search)")
    parser.add_argument("--tau",       type=float, default=0.60, help="Alert confidence threshold")
    parser.add_argument("--rate",      type=float, default=3.0,  help="Simulated flow rate (flows/sec)")
    parser.add_argument("--duration",  type=int,   default=None, help="Inference duration in seconds")
    args = parser.parse_args()

    # Default: full pipeline
    run_all = not (args.train or args.infer or args.dashboard)

    stage1_check()

    if args.train or run_all:
        data = stage2_preprocess()
        stage3_train(data=data, quick=args.quick)

    if args.infer:
        stage4_infer(tau=args.tau, rate=args.rate, duration=args.duration)
        return

    if args.dashboard or run_all:
        stage5_dashboard()


if __name__ == "__main__":
    main()
