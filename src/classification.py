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
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score, roc_curve, auc)
from xgboost import XGBClassifier

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')

# Severity 1=Fatal, 2=Serious, 3=Slight
SEVERITY_LABELS = ['Fatal', 'Serious', 'Slight']
CLASSES = [1, 2, 3]


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1 — LOAD
    # ------------------------------------------------------------------
    print("\n--- STEP 1: Load ---")
    X = pd.read_csv(os.path.join(OUTPUT_DIR, 'X_final.csv'))
    y = pd.read_csv(os.path.join(OUTPUT_DIR, 'y_final.csv')).squeeze()
    print(f"X shape: {X.shape} | y shape: {y.shape}")
    print(f"\nClass distribution:")
    counts = y.value_counts().sort_index()
    for cls, label in zip(CLASSES, SEVERITY_LABELS):
        n = counts.get(cls, 0)
        print(f"  Class {cls} ({label}): {n:,}  ({n/len(y)*100:.1f}%)")

    # ------------------------------------------------------------------
    # STEP 2 — TRAIN / TEST SPLIT
    # ------------------------------------------------------------------
    print("\n--- STEP 2: Train/Test Split (80/20) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    # ------------------------------------------------------------------
    # STEP 3 — CLASS IMBALANCE STRATEGY
    # ------------------------------------------------------------------
    # SMOTE is NOT used here. Most features are label-encoded categoricals
    # (Road_Type, Light_Conditions, Junction_Detail etc). SMOTE interpolates
    # between rows to create synthetic points — for categoricals this
    # produces values like Road_Type=1.5 which represent no real category.
    #
    # Instead we use class_weight='balanced' in each classifier.
    # This mathematically increases the penalty for misclassifying Fatal
    # and Serious accidents proportional to how rare they are — same effect
    # as oversampling but without creating any fake data.
    #
    # For XGBoost (no class_weight param): compute_sample_weight is used,
    # which assigns a higher weight to each Fatal/Serious training row.
    # ------------------------------------------------------------------
    print("\n--- STEP 3: Class Imbalance — Using class_weight='balanced' ---")
    print("No synthetic data created. Classifiers penalize minority class errors.")

    # Precompute sample weights for XGBoost
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    print(f"Sample weight range: min={sample_weights.min():.4f}  max={sample_weights.max():.4f}")

    # ------------------------------------------------------------------
    # STEP 4 — SCALE
    # ------------------------------------------------------------------
    # Fit scaler on real training data only (no SMOTE inflation)
    print("\n--- STEP 4: Scale ---")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # STEP 5 — BASELINE: LOGISTIC REGRESSION
    # ------------------------------------------------------------------
    print("\n--- STEP 5: Logistic Regression (Baseline) ---")
    lr = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',   # handles imbalance
        random_state=42,
        n_jobs=-1
    )
    lr.fit(X_train_sc, y_train)
    y_pred_lr    = lr.predict(X_test_sc)
    lr_test_acc  = accuracy_score(y_test, y_pred_lr)
    lr_train_acc = accuracy_score(y_train, lr.predict(X_train_sc))
    lr_f1        = f1_score(y_test, y_pred_lr, average='macro')
    print(classification_report(y_test, y_pred_lr, target_names=SEVERITY_LABELS))

    # ------------------------------------------------------------------
    # STEP 6 — RANDOM FOREST
    # ------------------------------------------------------------------
    print("\n--- STEP 6: Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',   # handles imbalance
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_sc, y_train)
    y_pred_rf    = rf.predict(X_test_sc)
    rf_test_acc  = accuracy_score(y_test, y_pred_rf)
    rf_train_acc = accuracy_score(y_train, rf.predict(X_train_sc))
    rf_f1        = f1_score(y_test, y_pred_rf, average='macro')
    print(classification_report(y_test, y_pred_rf, target_names=SEVERITY_LABELS))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

    # ------------------------------------------------------------------
    # STEP 7 — XGBOOST
    # ------------------------------------------------------------------
    # XGBoost requires 0-based integer labels so remap: 1->0, 2->1, 3->2
    # sample_weight is used instead of class_weight (not natively supported)
    # Predictions are remapped back to 1,2,3 for consistent reporting
    # ------------------------------------------------------------------
    print("\n--- STEP 7: XGBoost ---")
    y_train_xgb = y_train - 1   # 0-based for XGBoost
    y_test_xgb  = y_test  - 1

    xgb = XGBClassifier(
        n_estimators=200,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train_sc, y_train_xgb, sample_weight=sample_weights)

    y_pred_xgb_raw = xgb.predict(X_test_sc)
    y_pred_xgb     = y_pred_xgb_raw + 1   # remap back to 1, 2, 3

    # Train accuracy also remapped correctly
    xgb_train_acc = accuracy_score(y_train_xgb, xgb.predict(X_train_sc))
    xgb_test_acc  = accuracy_score(y_test, y_pred_xgb)
    xgb_f1        = f1_score(y_test, y_pred_xgb, average='macro')

    print(classification_report(y_test, y_pred_xgb, target_names=SEVERITY_LABELS))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

    # ------------------------------------------------------------------
    # STEP 8 — COMPARE MODELS
    # ------------------------------------------------------------------
    print("\n--- STEP 8: Model Comparison ---")
    results = {
        'Logistic Regression': {
            'model': lr, 'y_pred': y_pred_lr,
            'train_acc': lr_train_acc, 'test_acc': lr_test_acc, 'f1': lr_f1
        },
        'Random Forest': {
            'model': rf, 'y_pred': y_pred_rf,
            'train_acc': rf_train_acc, 'test_acc': rf_test_acc, 'f1': rf_f1
        },
        'XGBoost': {
            'model': xgb, 'y_pred': y_pred_xgb,
            'train_acc': xgb_train_acc, 'test_acc': xgb_test_acc, 'f1': xgb_f1
        },
    }

    print(f"\n{'Model':<22} {'Train Acc':>10} {'Test Acc':>10} {'F1-Macro':>10}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:<22} {r['train_acc']:>10.4f} {r['test_acc']:>10.4f} {r['f1']:>10.4f}")

    best_name = max(results, key=lambda k: results[k]['f1'])
    best      = results[best_name]
    print(f"\nBest model by F1-Macro: {best_name} ({best['f1']:.4f})")

    # ------------------------------------------------------------------
    # STEP 9 — CONFUSION MATRIX HEATMAP (best model)
    # ------------------------------------------------------------------
    print("\n--- STEP 9: Confusion Matrix Heatmap ---")
    cm = confusion_matrix(y_test, best['y_pred'], labels=CLASSES)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=SEVERITY_LABELS,
                yticklabels=SEVERITY_LABELS)
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

    # Binarize y_test using original 1,2,3 labels
    y_test_bin = label_binarize(y_test, classes=CLASSES)

    # Get probability estimates aligned to CLASSES order [1, 2, 3]
    if best_name == 'XGBoost':
        # XGBoost predicts probabilities for 0-based classes [0,1,2]
        # which map directly to [Fatal, Serious, Slight] — order is preserved
        y_prob = xgb.predict_proba(X_test_sc)
    else:
        y_prob = best['model'].predict_proba(X_test_sc)

    plt.figure(figsize=(9, 7))
    colors = ['red', 'orange', 'steelblue']
    for i, (label, color) in enumerate(zip(SEVERITY_LABELS, colors)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{label} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves (One-vs-Rest) — {best_name}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves.png'))
    plt.close()
    print("Saved roc_curves.png")

    # ------------------------------------------------------------------
    # STEP 11 — ACCURACY VS ESTIMATORS (Random Forest)
    # ------------------------------------------------------------------
    # Training 20 full RF models on 4.4M rows is very slow.
    # A stratified 10% subsample is used for this diagnostic plot only —
    # the actual classification above always uses the full training set.
    # ------------------------------------------------------------------
    print("\n--- STEP 11: Accuracy vs Estimators (Random Forest, 10% subsample) ---")

    # Subsample 10% stratified — for plot speed only
    X_sub, _, y_sub, _ = train_test_split(
        X_train_sc, y_train,
        test_size=0.90, random_state=42, stratify=y_train
    )

    train_accs, test_accs = [], []
    estimator_range = range(10, 210, 10)

    for n in estimator_range:
        rf_n = RandomForestClassifier(
            n_estimators=n,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_n.fit(X_sub, y_sub)
        train_accs.append(accuracy_score(y_sub,  rf_n.predict(X_sub)))
        test_accs.append(accuracy_score(y_test,  rf_n.predict(X_test_sc)))
        print(f"  n_estimators={n:>3}: train={train_accs[-1]:.4f}  test={test_accs[-1]:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(list(estimator_range), train_accs,
             label='Train Accuracy (10% subsample)', marker='o', markersize=4)
    plt.plot(list(estimator_range), test_accs,
             label='Test Accuracy (full test set)',  marker='s', markersize=4)
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
        f.write("Imbalance strategy: class_weight='balanced' (no SMOTE)\n")
        f.write("Reason: label-encoded categoricals make SMOTE interpolation invalid\n\n")
        f.write(f"{'Model':<22} {'Train Acc':>10} {'Test Acc':>10} {'F1-Macro':>10}\n")
        f.write("-" * 55 + "\n")
        for name, r in results.items():
            f.write(f"{name:<22} {r['train_acc']:>10.4f}"
                    f" {r['test_acc']:>10.4f} {r['f1']:>10.4f}\n")
        f.write(f"\nBest Model: {best_name}\n")
        f.write(f"  Test Accuracy : {best['test_acc']:.4f}\n")
        f.write(f"  F1-Macro      : {best['f1']:.4f}\n")
        f.write("\nFeatures used:\n")
        for col in X.columns:
            f.write(f"  - {col}\n")
    print("Saved classification_results.txt")

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("CLASSIFICATION COMPLETE")
    print("=" * 55)
    print(f"  Imbalance fix : class_weight='balanced' (no fake data)")
    print(f"  Best model    : {best_name}")
    print(f"  Test Accuracy : {best['test_acc']:.4f}")
    print(f"  F1-Macro      : {best['f1']:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    run()