"""
Stage 3 — Model Development
Trains a Random Forest classifier on NSL-KDD with grid search + 5-fold CV,
optimizing for macro F1-score (accounts for class imbalance).
Serializes model artifacts to model/artifacts/.
"""

import sys
import logging
import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

# Allow running as a script from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.preprocess import load_and_preprocess

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

# Grid search hyperparameter space (paper-guided)
PARAM_GRID = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [None, 20, 40],
    "min_samples_split": [2, 5, 10],
    "max_features":     ["sqrt", "log2"],
}


def train(
    X_train, y_train,
    X_test, y_test,
    feature_names,
    param_grid=None,
    n_cv_folds=5,
    artifacts_dir=ARTIFACTS_DIR,
    quick=False,          # quick=True uses a smaller grid for testing
):
    """
    Perform grid search with stratified k-fold CV and save the best model.

    Returns:
        best_model: fitted RandomForestClassifier
        metrics:    dict of evaluation metrics on the test set
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if quick:
        grid = {"n_estimators": [100], "max_depth": [20],
                "min_samples_split": [2], "max_features": ["sqrt"]}
    else:
        grid = param_grid or PARAM_GRID

    logger.info(f"[Train] Starting grid search over {grid} with {n_cv_folds}-fold CV ...")
    cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)

    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")
    gs = GridSearchCV(
        rf, grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=2,
        refit=True,
    )
    
    # Strictly enforce numpy arrays to prevent PyArrow backend crash in pandas 2.0+
    X_train_np = np.asarray(X_train, dtype=np.float32)
    y_train_np = np.asarray(y_train)
    
    gs.fit(X_train_np, y_train_np)

    best_model = gs.best_estimator_
    logger.info(f"[Train] Best params: {gs.best_params_}")
    logger.info(f"[Train] Best CV F1 (macro): {gs.best_score_:.4f}")

    # ── Evaluate on held-out test set ────────────────────────────────────────
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_).tolist()

    metrics = {
        "accuracy":        round(acc, 4),
        "f1_macro":        round(f1, 4),
        "best_cv_f1":      round(gs.best_score_, 4),
        "best_params":     gs.best_params_,
        "classification_report": report,
        "confusion_matrix": cm,
        "classes":         list(best_model.classes_),
    }

    logger.info(f"[Eval] Test accuracy : {acc:.4f}")
    logger.info(f"[Eval] Test F1 (macro): {f1:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred))

    # ── Feature importances ───────────────────────────────────────────────────
    importances = dict(zip(feature_names, best_model.feature_importances_.tolist()))
    metrics["feature_importances"] = dict(
        sorted(importances.items(), key=lambda x: x[1], reverse=True)
    )

    # ── Persist artifacts ─────────────────────────────────────────────────────
    model_path   = artifacts_dir / "random_forest.pkl"
    metrics_path = artifacts_dir / "metrics.json"

    joblib.dump(best_model, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(f"[Persist] Model saved to {model_path}")
    logger.info(f"[Persist] Metrics saved to {metrics_path}")

    return best_model, metrics


def load_model(artifacts_dir=ARTIFACTS_DIR):
    """Load the serialized Random Forest model."""
    path = artifacts_dir / "random_forest.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found at {path}. Run train.py first."
        )
    return joblib.load(path)


def load_metrics(artifacts_dir=ARTIFACTS_DIR):
    """Load the saved training metrics."""
    path = artifacts_dir / "metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(ARTIFACTS_DIR.parent / "training.log", mode="w"),
        ]
    )

    data = load_and_preprocess()
    model, metrics = train(
        data["X_train"], data["y_train"],
        data["X_test"],  data["y_test"],
        feature_names=data["feature_names"],
    )
    print("\n=== Final Test Metrics ===")
    print(f"  Accuracy  : {metrics['accuracy']}")
    print(f"  F1 (macro): {metrics['f1_macro']}")
    print(f"\nTop-10 features by importance:")
    for feat, imp in list(metrics["feature_importances"].items())[:10]:
        print(f"  {feat:40s} {imp:.4f}")
