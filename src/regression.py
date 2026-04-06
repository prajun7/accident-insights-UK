import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def print_metrics(name, y_true, y_pred):
    r = rmse(y_true, y_pred)
    m = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  RMSE : {r:.4f}")
    print(f"  MAE  : {m:.4f}")
    print(f"  R²   : {r2:.4f}")
    return r, m, r2


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1 — LOAD
    # ------------------------------------------------------------------
    print("\n--- STEP 1: Load ---")
    X_reg = pd.read_csv(os.path.join(OUTPUT_DIR, 'X_final.csv'))
    accidents = pd.read_csv(os.path.join(OUTPUT_DIR, 'accidents_cleaned.csv'),
                            low_memory=False)

    # Align by index and extract target
    X_reg = X_reg.reset_index(drop=True)
    y_reg = accidents['Number_of_Casualties'].reset_index(drop=True)

    # Ensure same length
    min_len = min(len(X_reg), len(y_reg))
    X_reg = X_reg.iloc[:min_len]
    y_reg = y_reg.iloc[:min_len]

    print(f"X_reg shape : {X_reg.shape}")
    print(f"y_reg shape : {y_reg.shape}")
    print(f"Target stats:\n  mean={y_reg.mean():.3f}  median={y_reg.median():.1f}  "
          f"min={y_reg.min()}  max={y_reg.max()}")

    # ------------------------------------------------------------------
    # STEP 2 — TRAIN / TEST SPLIT + SCALE
    # ------------------------------------------------------------------
    print("\n--- STEP 2: Train/Test Split (80/20) + Scale ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    # ------------------------------------------------------------------
    # STEP 3 — BASELINE: LINEAR REGRESSION
    # ------------------------------------------------------------------
    print("\n--- STEP 3: Linear Regression (Baseline) ---")
    lr = LinearRegression()
    lr.fit(X_train_sc, y_train)
    y_pred_lr = lr.predict(X_test_sc)
    lr_rmse, lr_mae, lr_r2 = print_metrics('Linear Regression', y_test, y_pred_lr)

    # ------------------------------------------------------------------
    # STEP 4 — RANDOM FOREST REGRESSOR
    # ------------------------------------------------------------------
    print("\n--- STEP 4: Random Forest Regressor ---")
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train_sc, y_train)
    y_pred_rf = rf.predict(X_test_sc)
    rf_rmse, rf_mae, rf_r2 = print_metrics('Random Forest', y_test, y_pred_rf)

    # ------------------------------------------------------------------
    # STEP 5 — XGBOOST REGRESSOR
    # ------------------------------------------------------------------
    print("\n--- STEP 5: XGBoost Regressor ---")
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb.fit(X_train_sc, y_train)
    y_pred_xgb = xgb.predict(X_test_sc)
    xgb_rmse, xgb_mae, xgb_r2 = print_metrics('XGBoost', y_test, y_pred_xgb)

    # ------------------------------------------------------------------
    # STEP 6 — COMPARISON TABLE
    # ------------------------------------------------------------------
    print("\n--- STEP 6: Model Comparison ---")
    results = {
        'Linear Regression': {'y_pred': y_pred_lr, 'rmse': lr_rmse,  'mae': lr_mae,  'r2': lr_r2},
        'Random Forest':     {'y_pred': y_pred_rf, 'rmse': rf_rmse,  'mae': rf_mae,  'r2': rf_r2},
        'XGBoost':           {'y_pred': y_pred_xgb,'rmse': xgb_rmse, 'mae': xgb_mae, 'r2': xgb_r2},
    }

    print(f"\n{'Model':<22} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print("-" * 50)
    for name, r in results.items():
        print(f"{name:<22} {r['rmse']:>8.4f} {r['mae']:>8.4f} {r['r2']:>8.4f}")

    best_name = min(results, key=lambda k: results[k]['rmse'])
    best      = results[best_name]
    print(f"\nBest model (lowest RMSE): {best_name} (RMSE={best['rmse']:.4f})")

    # Save results
    results_path = os.path.join(OUTPUT_DIR, 'regression_results.txt')
    with open(results_path, 'w') as f:
        f.write("UK TRAFFIC ACCIDENT — REGRESSION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Target variable: Number_of_Casualties\n\n")
        f.write(f"{'Model':<22} {'RMSE':>8} {'MAE':>8} {'R²':>8}\n")
        f.write("-" * 50 + "\n")
        for name, r in results.items():
            f.write(f"{name:<22} {r['rmse']:>8.4f} {r['mae']:>8.4f} {r['r2']:>8.4f}\n")
        f.write(f"\nBest Model: {best_name}\n")
        f.write(f"  RMSE : {best['rmse']:.4f}\n")
        f.write(f"  MAE  : {best['mae']:.4f}\n")
        f.write(f"  R²   : {best['r2']:.4f}\n")
    print("Saved regression_results.txt")

    # ------------------------------------------------------------------
    # STEP 7 — ACTUAL VS PREDICTED (best model)
    # ------------------------------------------------------------------
    print("\n--- STEP 7: Actual vs Predicted Plot ---")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best['y_pred'], alpha=0.4, s=10, color='steelblue', label='Predictions')
    lims = [min(y_test.min(), best['y_pred'].min()),
            max(y_test.max(), best['y_pred'].max())]
    plt.plot(lims, lims, 'r-', linewidth=2, label='Perfect prediction (y=x)')
    plt.xlabel('Actual Number of Casualties')
    plt.ylabel('Predicted Number of Casualties')
    plt.title(f'Actual vs Predicted — Number of Casualties\n({best_name})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'regression_actual_vs_predicted.png'))
    plt.close()
    print("Saved regression_actual_vs_predicted.png")

    # ------------------------------------------------------------------
    # STEP 8 — RESIDUALS HISTOGRAM (best model)
    # ------------------------------------------------------------------
    print("\n--- STEP 8: Residuals Histogram ---")
    residuals = best['y_pred'] - y_test
    plt.figure(figsize=(9, 6))
    plt.hist(residuals, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    plt.axvline(0, color='red', linewidth=2, linestyle='--', label='Zero residual')
    plt.xlabel('Residual (Predicted − Actual)')
    plt.ylabel('Count')
    plt.title(f'Residual Distribution — {best_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'residuals_histogram.png'))
    plt.close()
    print("Saved residuals_histogram.png")

    # ------------------------------------------------------------------
    # STEP 9 — LEARNING CURVE (best model, on subsample for speed)
    # ------------------------------------------------------------------
    print("\n--- STEP 9: Learning Curve ---")
    print("Computing learning curve — this may take a few minutes...")

    # Use a subsample to keep learning curve tractable
    SAMPLE = min(100_000, len(X_train_sc))
    idx = np.random.default_rng(42).choice(len(X_train_sc), size=SAMPLE, replace=False)
    X_lc = X_train_sc[idx]
    y_lc = y_train.iloc[idx].values

    if best_name == 'XGBoost':
        lc_model = XGBRegressor(n_estimators=100, learning_rate=0.1,
                                random_state=42, n_jobs=-1)
    elif best_name == 'Random Forest':
        lc_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        lc_model = LinearRegression()

    train_sizes, train_scores, val_scores = learning_curve(
        lc_model, X_lc, y_lc,
        train_sizes=np.linspace(0.1, 1.0, 8),
        scoring='neg_root_mean_squared_error',
        cv=3, n_jobs=-1
    )

    train_rmse = -train_scores.mean(axis=1)
    val_rmse   = -val_scores.mean(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_rmse, 'o-', label='Train RMSE', color='steelblue')
    plt.plot(train_sizes, val_rmse,   's-', label='Validation RMSE', color='orange')
    plt.xlabel('Training Set Size')
    plt.ylabel('RMSE')
    plt.title(f'Learning Curve — {best_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'learning_curve.png'))
    plt.close()
    print("Saved learning_curve.png")

    print("\n" + "=" * 50)
    print("REGRESSION COMPLETE")
    print("=" * 50)
    print(f"  Target        : Number_of_Casualties")
    print(f"  Best model    : {best_name}")
    print(f"  RMSE          : {best['rmse']:.4f}")
    print(f"  MAE           : {best['mae']:.4f}")
    print(f"  R²            : {best['r2']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    run()
