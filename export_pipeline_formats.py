# export_pipeline_formats.py
import joblib
import pickle
import os
import sys

SRC_JOBLIB = "laptop_performance_pipeline.joblib"
OUT_PKL = "laptop_performance_pipeline.pkl"

if not os.path.exists(SRC_JOBLIB):
    print(f"Could not find {SRC_JOBLIB}. Run train_laptop_perf.py first.")
    sys.exit(1)

pipeline = joblib.load(SRC_JOBLIB)
joblib.dump(pipeline, SRC_JOBLIB)  # overwrite safe
print(f"Saved joblib -> {SRC_JOBLIB}")

with open(OUT_PKL, "wb") as f:
    pickle.dump(pipeline, f)
print(f"Saved pickle -> {OUT_PKL}")

print("Done. Use joblib for Streamlit deployment (recommended).")
