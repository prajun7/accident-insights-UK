import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score, roc_curve, auc)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')

SEVERITY_LABELS = ['Fatal', 'Serious', 'Slight']   # maps to classes 1, 2, 3


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1 — LOAD
    # ------------------------------------------------------------------
    print("\n--- STEP 1: Load ---")
    X = pd.read_csv(os.path.join(OUTPUT_DIR, 'X_final.csv'))
    y = pd.read_csv(os.path.join(OUTPUT_DIR, 'y_final.csv')).squeeze()
    print(f"X shape: {X.shape} | y shape: {y.shape}")
    print(f"Class distribution:\n{y.value_counts().sort_index()}")

    # ------------------------------------------------------------------
    # STEP 2 — TRAIN / TEST SPLIT
    # ------------------------------------------------------------------
    print("\n--- STEP 2: Train/Test Split (80/20) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape} | y_test: {y_test.shape}")

    # ------------------------------------------------------------------
    # STEP 3 — SMOTE (training set only)
    # ------------------------------------------------------------------
    print("\n--- STEP 3: SMOTE ---")
    print(f"Class distribution before SMOTE:\n{y_train.value_counts().sort_index()}")
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"Class distribution after SMOTE:\n{pd.Series(y_train_sm).value_counts().sort_index()}")
    print(f"Training set size after SMOTE: {X_train_sm.shape}")

    # ------------------------------------------------------------------
    # STEP 4 — SCALE
    # ------------------------------------------------------------------
    print("\n--- STEP 4: Scale ---")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_sm)
    X_test_sc  = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # STEP 5 — BASELINE: LOGISTIC REGRESSION
    # ------------------------------------------------------------------
    print("\n--- STEP 5: Logistic Regression (Baseline) ---")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train_sm)
    y_pred_lr = lr.predict(X_test_sc)
    print(classification_report(y_test, y_pred_lr, target_names=SEVERITY_LABELS))
    lr_acc    = accuracy_score(y_test, y_pred_lr)
    lr_f1     = f1_score(y_test, y_pred_lr, average='macro')
    lr_train_acc = accuracy_score(y_train_sm, lr.predict(X_train_sc))

    # ------------------------------------------------------------------
    # STEP 6 — RANDOM FOREST
    # ------------------------------------------------------------------
    print("\n--- STEP 6: Random Forest ---")
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                random_state=42, n_jobs=-1)
    rf.fit(X_train_sc, y_train_sm)
    y_pred_rf = rf.predict(X_test_sc)
    print(classification_report(y_test, y_pred_rf, target_names=SEVERITY_LABELS))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
    rf_acc    = accuracy_score(y_test, y_pred_rf)
    rf_f1     = f1_score(y_test, y_pred_rf, average='macro')
    rf_train_acc = accuracy_score(y_train_sm, rf.predict(X_train_sc))

    # ------------------------------------------------------------------
    # STEP 7 — XGBOOST (y remapped to 0-based)
    # ------------------------------------------------------------------
    print("\n--- STEP 7: XGBoost ---")
    y_train_xgb = y_train_sm - 1
    y_test_xgb  = y_test - 1

    xgb = XGBClassifier(n_estimators=200, eval_metric='mlogloss',
                        random_state=42, n_jobs=-1)
    xgb.fit(X_train_sc, y_train_xgb)
    y_pred_xgb_raw = xgb.predict(X_test_sc)
    y_pred_xgb = y_pred_xgb_raw + 1   # remap back to 1,2,3

    print(classification_report(y_test, y_pred_xgb, target_names=SEVERITY_LABELS))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
    xgb_acc    = accuracy_score(y_test, y_pred_xgb)
    xgb_f1     = f1_score(y_test, y_pred_xgb, average='macro')
    xgb_train_acc = accuracy_score(y_train_xgb, xgb.predict(X_train_sc))

    # ------------------------------------------------------------------
    # STEP 8 — COMPARE MODELS
    # ------------------------------------------------------------------
    print("\n--- STEP 8: Model Comparison ---")
    results = {
        'Logistic Regression': {'model': lr, 'y_pred': y_pred_lr,
                                'train_acc': lr_train_acc,  'test_acc': lr_acc,  'f1': lr_f1},
        'Random Forest':       {'model': rf, 'y_pred': y_pred_rf,
                                'train_acc': rf_train_acc,  'test_acc': rf_acc,  'f1': rf_f1},
        'XGBoost':             {'model': xgb,'y_pred': y_pred_xgb,
                                'train_acc': xgb_train_acc, 'test_acc': xgb_acc, 'f1': xgb_f1},
    }

    print(f"\n{'Model':<22} {'Train Acc':>10} {'Test Acc':>10} {'F1-Macro':>10}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:<22} {r['train_acc']:>10.4f} {r['test_acc']:>10.4f} {r['f1']:>10.4f}")

    best_name = max(results, key=lambda k: results[k]['f1'])
    best      = results[best_name]
    print(f"\nBest model: {best_name} (F1-Macro={best['f1']:.4f})")

    # ------------------------------------------------------------------
    # STEP 9 — CONFUSION MATRIX HEATMAP (best model)
    # ------------------------------------------------------------------
    print("\n--- STEP 9: Confusion Matrix Heatmap ---")
    cm = confusion_matrix(y_test, best['y_pred'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=SEVERITY_LABELS, yticklabels=SEVERITY_LABELS)
    plt.title(f'Confusion Matrix — {best_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()
    print("Saved confusion_matrix.png")

    # ------------------------------------------------------------------
    # STEP 10 — ROC CURVES (best model, one-vs-rest)
    # ------------------------------------------------------------------
    print("\n--- STEP 10: ROC Curves ---")
    classes = sorted(y.unique())   # [1, 2, 3]
    y_test_bin = label_binarize(y_test, classes=classes)

    # Get probability estimates
    if best_name == 'XGBoost':
        y_prob = xgb.predict_proba(X_test_sc)
    else:
        y_prob = best['model'].predict_proba(X_test_sc)

    plt.figure(figsize=(9, 7))
    colors = ['red', 'orange', 'steelblue']
    for i, (cls, label, color) in enumerate(zip(classes, SEVERITY_LABELS, colors)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{label} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves — {best_name}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves.png'))
    plt.close()
    print("Saved roc_curves.png")

    # ------------------------------------------------------------------
    # STEP 11 — ACCURACY VS ESTIMATORS (Random Forest)
    # ------------------------------------------------------------------
    print("\n--- STEP 11: Accuracy vs Estimators (Random Forest) ---")
    train_accs, test_accs, estimator_range = [], [], range(10, 210, 10)
    for n in estimator_range:
        rf_n = RandomForestClassifier(n_estimators=n, class_weight='balanced',
                                      random_state=42, n_jobs=-1)
        rf_n.fit(X_train_sc, y_train_sm)
        train_accs.append(accuracy_score(y_train_sm, rf_n.predict(X_train_sc)))
        test_accs.append(accuracy_score(y_test, rf_n.predict(X_test_sc)))
        print(f"  n_estimators={n:>3}: train={train_accs[-1]:.4f}  test={test_accs[-1]:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(list(estimator_range), train_accs, label='Train Accuracy', marker='o', markersize=4)
    plt.plot(list(estimator_range), test_accs,  label='Test Accuracy',  marker='s', markersize=4)
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.title('Random Forest: Accuracy vs Number of Estimators')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_vs_estimators.png'))
    plt.close()
    print("Saved accuracy_vs_estimators.png")

    # ------------------------------------------------------------------
    # STEP 12 — SAVE RESULTS
    # ------------------------------------------------------------------
    print("\n--- STEP 12: Save Results ---")
    results_path = os.path.join(OUTPUT_DIR, 'classification_results.txt')
    with open(results_path, 'w') as f:
        f.write("UK TRAFFIC ACCIDENT — CLASSIFICATION RESULTS\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"{'Model':<22} {'Train Acc':>10} {'Test Acc':>10} {'F1-Macro':>10}\n")
        f.write("-" * 55 + "\n")
        for name, r in results.items():
            f.write(f"{name:<22} {r['train_acc']:>10.4f} {r['test_acc']:>10.4f} {r['f1']:>10.4f}\n")
        f.write(f"\nBest Model: {best_name}\n")
        f.write(f"  Test Accuracy : {best['test_acc']:.4f}\n")
        f.write(f"  F1-Macro      : {best['f1']:.4f}\n")
        f.write("\nFeatures used:\n")
        for col in X.columns:
            f.write(f"  - {col}\n")
    print(f"Saved classification_results.txt")

    print("\n" + "=" * 55)
    print("CLASSIFICATION COMPLETE")
    print("=" * 55)
    print(f"  Best model  : {best_name}")
    print(f"  Test Acc    : {best['test_acc']:.4f}")
    print(f"  F1-Macro    : {best['f1']:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    run()
