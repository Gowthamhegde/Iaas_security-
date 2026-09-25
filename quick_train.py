"""
Quick-train script for testing the full pipeline end-to-end.
Uses a single RF configuration (no grid search) for speed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import stage1_check, stage2_preprocess, stage3_train

if __name__ == "__main__":
    stage1_check()
    data = stage2_preprocess()
    stage3_train(data=data, quick=True)
    print("\nQuick train complete. Run the dashboard with:")
    print("  streamlit run dashboard/app.py")
