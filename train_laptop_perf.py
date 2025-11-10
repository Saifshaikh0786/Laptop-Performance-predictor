"""
train_laptop_perf.py

Generates synthetic laptop hardware dataset, trains regression models to predict a
performance_score, optionally runs a randomized hyperparameter search, evaluates,
and saves the final pipeline.

Adjust N_SAMPLES, DO_TUNE, N_ITER_SEARCH as needed.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")

def generate_synthetic_laptop_data(n_samples=5000, random_state=42):
    rng = np.random.RandomState(random_state)
    cpu_brands = ['Intel', 'AMD', 'Apple']
    cpu_tiers = ['low', 'mid', 'high']
    gpu_types = ['Integrated', 'NVIDIA GTX', 'NVIDIA RTX', 'AMD Radeon']
    storage_types = ['HDD', 'SATA SSD', 'NVMe SSD']
    resolutions = ['1366x768', '1920x1080', '2560x1440', '3840x2160']
    manufacturers = ['Dell', 'HP', 'Lenovo', 'Asus', 'Acer', 'Apple', 'MSI', 'Razer']

    cpu_brand = rng.choice(cpu_brands, n_samples, p=[0.45, 0.45, 0.10])
    cpu_tier = rng.choice(cpu_tiers, n_samples, p=[0.35, 0.45, 0.20])
    gpu_type = rng.choice(gpu_types, n_samples, p=[0.40, 0.25, 0.20, 0.15])
    storage_type = rng.choice(storage_types, n_samples, p=[0.10, 0.35, 0.55])
    resolution = rng.choice(resolutions, n_samples, p=[0.05, 0.70, 0.18, 0.07])
    manufacturer = rng.choice(manufacturers, n_samples)

    year = rng.choice(np.arange(2016, 2026), n_samples, p=np.linspace(0.5, 1.5, 10)/np.linspace(0.5, 1.5, 10).sum())

    cores = []
    base_clock = []
    boost_clock = []
    cpu_score = []
    for b, t, y in zip(cpu_brand, cpu_tier, year):
        if t == 'low':
            c = rng.choice([2, 4], p=[0.3, 0.7])
            bc = float(round(rng.normal(2.3, 0.25), 2))
            boost = float(round(bc + rng.uniform(0.2, 0.7), 2))
        elif t == 'mid':
            c = rng.choice([4, 6, 8], p=[0.2, 0.6, 0.2])
            bc = float(round(rng.normal(2.7, 0.3), 2))
            boost = float(round(bc + rng.uniform(0.4, 1.0), 2))
        else:
            c = rng.choice([6, 8, 12, 16], p=[0.15, 0.5, 0.25, 0.10])
            bc = float(round(rng.normal(2.9, 0.3), 2))
            boost = float(round(bc + rng.uniform(0.6, 1.6), 2))

        cores.append(int(c))
        base_clock.append(bc)
        boost_clock.append(boost)
        brand_multiplier = 1.0
        if b == 'Apple':
            brand_multiplier = 1.15
        elif b == 'AMD':
            brand_multiplier = 1.03
        tier_mul = {'low': 0.7, 'mid': 1.0, 'high': 1.35}[t]
        score = max(100, int((c * (bc*1.2 + boost*0.8) * tier_mul * brand_multiplier) * (1 + (y-2016)/20) + rng.normal(0, 30)))
        cpu_score.append(score)

    gpu_score = []
    for g, y in zip(gpu_type, year):
        if g == 'Integrated':
            base = rng.normal(60, 20)
        elif g == 'NVIDIA GTX':
            base = rng.normal(400, 80)
        elif g == 'NVIDIA RTX':
            base = rng.normal(900, 200)
        else:
            base = rng.normal(450, 120)
        gpu_score.append(int(max(20, base * (1 + (y-2018)/10) + rng.normal(0, 40))))

    ram_gb = []
    for t in cpu_tier:
        if t == 'low':
            ram_gb.append(rng.choice([4, 8], p=[0.25, 0.75]))
        elif t == 'mid':
            ram_gb.append(rng.choice([8, 16], p=[0.35, 0.65]))
        else:
            ram_gb.append(rng.choice([16, 32, 64], p=[0.6, 0.35, 0.05]))
    ram_speed = rng.choice([2133, 2400, 2666, 3000, 3200, 3600, 4266, 4800], n_samples, p=[0.05,0.1,0.2,0.2,0.2,0.15,0.06,0.04])

    storage_gb = rng.choice([128, 256, 512, 1024, 2048], n_samples, p=[0.15,0.35,0.3,0.15,0.05])
    storage_speed = []
    for st in storage_type:
        if st == 'HDD':
            storage_speed.append(rng.normal(120, 20))
        elif st == 'SATA SSD':
            storage_speed.append(rng.normal(500, 50))
        else:
            storage_speed.append(rng.normal(3000, 600))

    display_in = rng.normal(15.6, 1.2, n_samples).round(1)
    weight_kg = (display_in/10.0) + rng.normal(0.4, 0.25, n_samples)
    battery_mah = rng.choice([3500, 4500, 6000, 8000], n_samples, p=[0.25, 0.35, 0.30, 0.10])

    tdp = []
    for t, g in zip(cpu_tier, gpu_type):
        base_tdp = 15 if t == 'low' else (28 if t == 'mid' else 45)
        if g in ['NVIDIA RTX', 'NVIDIA GTX', 'AMD Radeon']:
            base_tdp += 35
        tdp.append(int(rng.normal(base_tdp, 8)))

    price = []
    for cs, gs, r, stype, m in zip(cpu_score, gpu_score, ram_gb, storage_type, manufacturer):
        base_price = (cs*0.6 + gs*0.4) / 3.0
        if m == 'Apple': base_price *= 1.45
        if stype == 'NVMe SSD': base_price *= 1.08
        price.append(int(max(250, base_price + rng.normal(0, 80))))

    df = pd.DataFrame({
        'manufacturer': manufacturer,
        'year': year,
        'cpu_brand': cpu_brand,
        'cpu_tier': cpu_tier,
        'cores': cores,
        'base_clock': base_clock,
        'boost_clock': boost_clock,
        'cpu_score': cpu_score,
        'gpu_type': gpu_type,
        'gpu_score': gpu_score,
        'ram_gb': ram_gb,
        'ram_speed_mhz': ram_speed,
        'storage_type': storage_type,
        'storage_gb': storage_gb,
        'storage_speed_mb_s': storage_speed,
        'display_in': display_in,
        'resolution': resolution,
        'weight_kg': weight_kg.round(2),
        'battery_mah': battery_mah,
        'tdp_watt': tdp,
        'price_usd': price
    })

    perf = (
        0.45 * df['cpu_score'] +
        0.35 * df['gpu_score'] +
        0.07 * df['ram_gb'] * 25 +
        0.04 * df['storage_speed_mb_s'] / 10 +
        0.03 * (df['ram_speed_mhz'] / 100.0) +
        0.02 * (df['year'] - 2015) * 10 +
        0.04 * (df['price_usd']/100.0)
    )
    perf = perf - (df['weight_kg'] - df['display_in']/10.0) * 8 - (df['tdp_watt']-30).clip(lower=0) * 0.6
    perf = perf * (1 + 0.003 * (df['cpu_tier'].map({'low': -1, 'mid': 0, 'high': 1})))
    perf = perf + rng.normal(0, 40, n_samples)
    df['performance_score'] = perf.round(1).clip(lower=10)
    return df

def build_and_train(n_samples=5000, do_tune=False, n_iter_search=12, random_state=42):
    print(f"Generating {n_samples} synthetic rows...")
    df = generate_synthetic_laptop_data(n_samples=n_samples, random_state=random_state)

    target_col = 'performance_score'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)


    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)

    # Choose default model (fast & strong): HistGradientBoosting; if you prefer, switch to RandomForest.
    base_model = HistGradientBoostingRegressor(random_state=random_state)
    pipe = Pipeline(steps=[('pre', preprocessor), ('model', base_model)])

    if do_tune:
        print("Running RandomizedSearchCV (tuning) -- runtime depends on n_iter_search & data size...")
        param_dist = {
            'model__learning_rate': [0.01, 0.03, 0.05, 0.1],
            'model__max_iter': [100, 200, 400],
            'model__max_depth': [3, 6, 10, None],
            'model__min_samples_leaf': [20, 50, 100]
        }
        search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=n_iter_search,
                                    cv=3, scoring='r2', random_state=random_state, n_jobs=-1, verbose=1)
        start = time.time()
        search.fit(X_train, y_train)
        print(f"Tuning done in {time.time()-start:.1f}s. Best params:")
        print(search.best_params_)
        best_pipeline = search.best_estimator_
    else:
        print("Fitting default pipeline (no tuning)...")
        start = time.time()
        pipe.fit(X_train, y_train)
        print(f"Fitted in {time.time()-start:.1f}s")
        best_pipeline = pipe

    # Evaluate
    y_pred = best_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)


    print(f"Test performance: R2={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")

    # Save pipeline
    model_filename = 'laptop_performance_pipeline.joblib'
    joblib.dump(best_pipeline, model_filename)
    print(f"Saved pipeline to {model_filename}")

    # Try to show top feature importances if available
    try:
        model_step = best_pipeline.named_steps['model']
        if hasattr(model_step, 'feature_importances_'):
            importances = model_step.feature_importances_
            cat_ohe = best_pipeline.named_steps['pre'].named_transformers_['cat']
            ohe_names = list(cat_ohe.get_feature_names_out(categorical_features))
            feature_names = numeric_features + ohe_names
            fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(30)
            print("\nTop features by importance:")
            print(fi.head(15))
            plt.figure(figsize=(10,5))
            fi.head(20).plot.bar()
            plt.title("Top 20 feature importances")
            plt.tight_layout()
            plt.show()
        else:
            print("Model does not provide feature_importances_")
    except Exception as e:
        print("Could not compute/display feature importances:", e)

    # Save a small sample csv
    df.sample(min(500, len(df)), random_state=1).to_csv('synthetic_laptops_sample.csv', index=False)
    print("Saved sample CSV: synthetic_laptops_sample.csv")

    return best_pipeline, df

if __name__ == '__main__':
    # CONFIG - tune these
    N_SAMPLES = 5000         # increase to 10000 or 20000 for potentially better accuracy (more compute/time)
    DO_TUNE = False          # set True to run hyperparameter tuning (slower)
    N_ITER_SEARCH = 12       # number of iterations for RandomizedSearchCV (if DO_TUNE True)

    model_pipeline, full_df = build_and_train(n_samples=N_SAMPLES, do_tune=DO_TUNE, n_iter_search=N_ITER_SEARCH)

    # Example usage: predict on a new record (dict must include same feature names)
    sample = full_df.drop(columns=['performance_score']).iloc[0:3].to_dict(orient='records')
    preds = model_pipeline.predict(pd.DataFrame(sample))
    print("\nExample predictions for 3 sample rows:")
    print(preds)
