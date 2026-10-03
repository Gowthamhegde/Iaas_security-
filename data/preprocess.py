"""
Stage 2 — Data Acquisition & Preprocessing
Downloads NSL-KDD, cleans, encodes, and scales for model training.
Persists encoders/scaler so live inference uses the exact train-time transform.
"""

import logging
import requests
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent.parent / "model" / "artifacts"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"

# ── NSL-KDD column definitions ────────────────────────────────────────────────
FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]

# KDD attack -> 5-class mapping (as per Table III of the paper)
ATTACK_MAP = {
    "normal": "Normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS", "apache2": "DoS",
    "processtable": "DoS", "udpstorm": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "sendmail": "R2L", "named": "R2L", "snmpgetattack": "R2L",
    "snmpguess": "R2L", "worm": "R2L", "xlock": "R2L", "xsnoop": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "httptunnel": "U2R", "ps": "U2R",
    "sqlattack": "U2R", "xterm": "U2R",
    # CICIDS2017 mappings
    "benign": "Normal",
    "dos hulk": "DoS", "dos goldeneye": "DoS", "dos slowloris": "DoS", "dos slowhttptest": "DoS",
    "ddos": "DoS",
    "portscan": "Probe",
    "ftp-patator": "R2L", "ssh-patator": "R2L",
    "bot": "R2L",
    "web attack - brute force": "U2R", "web attack - xss": "U2R", "web attack - sql injection": "U2R",
    "infiltration": "U2R",
    "heartbleed": "Probe"
}

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# Public mirrors for NSL-KDD
NSLKDD_URLS = {
    
    "train": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt",
    "test":  "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt",
}

DATA_DIR = Path(__file__).parent / "raw"


def download_nslkdd(data_dir=DATA_DIR):
    """Download NSL-KDD train/test files if not already present."""
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, url in NSLKDD_URLS.items():
        dest = data_dir / f"KDD{split.capitalize()}+.txt"
        if dest.exists():
            logger.info(f"[NSL-KDD] {split} already cached at {dest}")
        else:
            logger.info(f"[NSL-KDD] Downloading {split} from {url} ...")
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            logger.info(f"[NSL-KDD] Saved {split} -> {dest} ({dest.stat().st_size:,} bytes)")
        paths[split] = dest
    return paths


def _load_raw(path):
    """Load a raw NSL-KDD .txt file into a DataFrame."""
    df = pd.read_csv(path, header=None, names=FEATURE_NAMES)
    df.drop(columns=["difficulty"], inplace=True)
    return df


def _map_labels(df):
    """Map raw KDD/CICIDS attack strings to 5-class labels."""
    df = df.copy()
    if "Label" in df.columns:
        df.rename(columns={"Label": "label"}, inplace=True)
    df["label"] = df["label"].astype(str).str.strip().str.lower().str.rstrip(".")
    df["label"] = df["label"].map(ATTACK_MAP).fillna("U2R")
    return df

def _load_cicids2017(data_dir=DATA_DIR):
    """Load CICIDS2017 data if available, or return a small dummy DataFrame."""
    cic_path = data_dir / "cicids2017.csv"
    if cic_path.exists():
        logger.info(f"[CICIDS2017] Loading dataset from {cic_path} ...")
        df = pd.read_csv(cic_path)
        # Standardize column names (strip spaces, lowercase)
        df.columns = df.columns.str.strip().str.lower()
        if "label" not in df.columns and "Label" in df.columns:
            df.rename(columns={"Label": "label"}, inplace=True)
        return df
    
    logger.warning("[CICIDS2017] cicids2017.csv not found in data/raw/. Using a dummy sample for the merge.")
    # Create a dummy dataframe with some CICIDS2017 specific columns to demonstrate the merge
    dummy_data = {
        "flow duration": [100, 200, 300, 400],
        "total fwd packets": [2, 3, 4, 5],
        "total backward packets": [1, 2, 3, 4],
        "label": ["BENIGN", "DoS Hulk", "PortScan", "Bot"]
    }
    return pd.DataFrame(dummy_data)


def _encode_categoricals(df, encoders=None, fit=True):
    """
    Label-encode categorical columns.
    Returns (encoded_df, encoders_dict).
    """
    df = df.copy()
    if encoders is None:
        encoders = {}
    for col in CATEGORICAL_COLS:
        le = encoders.get(col, LabelEncoder())
        if fit:
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x, k=known, c=le.classes_: x if x in k else c[0]
            )
            df[col] = le.transform(df[col])
    return df, encoders


def _scale_features(X, scaler=None, fit=True):
    """Standard-scale all feature columns."""
    if scaler is None:
        scaler = StandardScaler()
    if fit:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler


def load_and_preprocess(data_dir=DATA_DIR):
    """
    Full preprocessing pipeline:
      1. Download NSL-KDD if needed
      2. Load raw files
      3. Load CICIDS2017
      4. Map labels -> 5-class
      5. Merge datasets (filling missing features with 0)
      6. Encode categoricals
      7. Scale features
    Returns a dict with X_train, y_train, X_test, y_test, encoders, scaler, feature_names.
    """
    paths = download_nslkdd(data_dir)

    train_df = _load_raw(paths["train"])
    test_df  = _load_raw(paths["test"])
    
    # Load CICIDS2017
    cicids_df = _load_cicids2017(data_dir)

    train_df = _map_labels(train_df)
    test_df  = _map_labels(test_df)
    cicids_df = _map_labels(cicids_df)
    
    # Split CICIDS2017 into train/test to merge with NSL-KDD
    # Simple split 80/20 for the sake of the merge
    np.random.seed(42)
    mask = np.random.rand(len(cicids_df)) < 0.8
    cicids_train = cicids_df[mask]
    cicids_test = cicids_df[~mask]
    
    # Merge NSL-KDD and CICIDS2017
    # Concat will align common columns and place NaN in missing ones
    merged_train_df = pd.concat([train_df, cicids_train], ignore_index=True, sort=False)
    merged_test_df = pd.concat([test_df, cicids_test], ignore_index=True, sort=False)
    
    # Fill missing features: 0 for numeric, '0' for string
    for df in [merged_train_df, merged_test_df]:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna("0")

    y_train = merged_train_df["label"].values
    y_test  = merged_test_df["label"].values

    X_train = merged_train_df.drop(columns=["label"])
    X_test  = merged_test_df.drop(columns=["label"])

    X_train, encoders = _encode_categoricals(X_train, fit=True)
    X_test,  _        = _encode_categoricals(X_test, encoders=encoders, fit=False)

    # Some features might be non-numeric due to dirty data (e.g., 'Infinity' or 'NaN' in CICIDS2017)
    # We force convert to numeric and fill with 0
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.apply(pd.to_numeric, errors='coerce')
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    feature_names = list(X_train.columns)

    X_train_scaled, scaler = _scale_features(X_train, fit=True)
    X_test_scaled,  _      = _scale_features(X_test, scaler=scaler, fit=False)

    logger.info(
        f"[Preprocessing] Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}"
    )
    logger.info(f"[Preprocessing] Class distribution (train): "
                f"{pd.Series(y_train).value_counts().to_dict()}")

    result = {
        "X_train":       X_train_scaled,
        "y_train":       y_train,
        "X_test":        X_test_scaled,
        "y_test":        y_test,
        "encoders":      encoders,
        "scaler":        scaler,
        "feature_names": feature_names,
    }
    save_preprocessor(encoders, scaler, feature_names)
    return result


def save_preprocessor(encoders, scaler, feature_names, path=PREPROCESSOR_PATH):
    """Persist train-time encoders/scaler for live-server inference."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "encoders":      encoders,
            "scaler":        scaler,
            "feature_names": list(feature_names),
        },
        path,
    )
    logger.info(f"[Preprocessing] Saved preprocessor -> {path}")


def load_preprocessor(path=PREPROCESSOR_PATH):
    """
    Load persisted preprocessor for live inference.
    If missing, rebuild from NSL-KDD train split (deterministic LabelEncoder/Scaler).
    """
    path = Path(path)
    if path.exists():
        bundle = joblib.load(path)
        logger.info(f"[Preprocessing] Loaded preprocessor from {path}")
        return bundle

    logger.warning(
        "[Preprocessing] No preprocessor.joblib found — rebuilding from NSL-KDD train data."
    )
    data = load_and_preprocess()
    return {
        "encoders":      data["encoders"],
        "scaler":        data["scaler"],
        "feature_names": data["feature_names"],
    }


def preprocess_single_flow(flow, encoders, scaler, feature_names):
    """
    Transform a single raw flow dict into a scaled feature vector.
    Used by the inference service at runtime.
    """
    row = pd.DataFrame([flow])
    for col in feature_names:
        if col not in row.columns:
            row[col] = 0
    row = row[feature_names]
    row, _ = _encode_categoricals(row, encoders=encoders, fit=False)
    X, _ = _scale_features(row, scaler=scaler, fit=False)
    return X


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data = load_and_preprocess()
    print("Train shape:", data["X_train"].shape)
    print("Test shape: ", data["X_test"].shape)
    print("Classes:    ", sorted(set(data["y_train"])))
