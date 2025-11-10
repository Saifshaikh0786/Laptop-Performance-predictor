import argparse
import joblib
import pandas as pd
import os
import sys

MODEL_FILE = "laptop_performance_pipeline.joblib"

REQUIRED_FEATURES = [
    "manufacturer", "year", "cpu_brand", "cpu_tier",
    "cores", "base_clock", "boost_clock", "cpu_score",
    "gpu_type", "gpu_score", "ram_gb", "ram_speed_mhz",
    "storage_type", "storage_gb", "storage_speed_mb_s",
    "display_in", "resolution", "weight_kg", "battery_mah",
    "tdp_watt", "price_usd"
]

# ✅ Smart defaults (used if user does not give a parameter)
SMART_DEFAULTS = {
    "manufacturer": "Dell",
    "year": 2023,
    "cpu_brand": "Intel",
    "cpu_tier": "mid",
    "cores": 6,
    "base_clock": 2.7,
    "boost_clock": 4.0,
    "cpu_score": 650,
    "gpu_type": "Integrated",
    "gpu_score": 150,
    "ram_gb": 16,
    "ram_speed_mhz": 3200,
    "storage_type": "NVMe SSD",
    "storage_gb": 512,
    "storage_speed_mb_s": 2500,
    "display_in": 15.6,
    "resolution": "1920x1080",
    "weight_kg": 1.9,
    "battery_mah": 6000,
    "tdp_watt": 45,
    "price_usd": 900
}

# -----------------------------
# Load model
# -----------------------------
if not os.path.exists(MODEL_FILE):
    print("❌ ERROR: Could not find model:", MODEL_FILE)
    sys.exit(1)

model = joblib.load(MODEL_FILE)

# -----------------------------
# CLI Parser (all optional)
# -----------------------------
parser = argparse.ArgumentParser(description="Flexible Laptop Performance Predictor")

for feat in REQUIRED_FEATURES:
    parser.add_argument(f"--{feat}", help=f"{feat} (optional)")

args = parser.parse_args()

# -----------------------------
# Build data row
# -----------------------------
row = {}

for feat in REQUIRED_FEATURES:
    val = getattr(args, feat)
    if val is None:
        val = SMART_DEFAULTS[feat]   # ✅ fallback default

    # ✅ Type conversion
    try:
        if feat in ["year", "cores", "cpu_score", "gpu_score", "ram_gb",
                    "ram_speed_mhz", "storage_gb", "battery_mah", "tdp_watt"]:
            val = int(val)
        elif feat in ["base_clock", "boost_clock", "storage_speed_mb_s",
                      "display_in", "weight_kg", "price_usd"]:
            val = float(val)
    except:
        pass

    row[feat] = val

df = pd.DataFrame([row])

# -----------------------------
# Predict
# -----------------------------
pred = model.predict(df)[0]

# Normalize score out of 1000
normalized_score = min(max(pred / 1000, 0), 1) * 1000

# Performance Class
if pred > 700:
    category = "🔥 High Performance (Gaming / Editing / AI)"
elif pred > 350:
    category = "⚡ Mid Range (Coding / Office / Light Gaming)"
else:
    category = "📘 Entry Level (Office / Browsing)"

# Estimated years laptop will “run smoothly”
expected_years = round((pred / 100) + 3, 1)
expected_years = min(expected_years, 10)  # cap at 10 years

# -----------------------------
# DISPLAY OUTPUT
# -----------------------------
print("\n✅ FINAL INPUTS USED:")
print(df)

print("\n🎯 PREDICTION RESULT")
print(f"• Performance Score: {pred:.2f} / 1000")
print(f"• Normalized Score: {normalized_score:.1f} / 1000")
print(f"• Category: {category}")
print(f"• Expected Smooth Years: {expected_years} years\n")
