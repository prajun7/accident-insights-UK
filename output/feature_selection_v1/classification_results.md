# Classification Results

## Input
- **X_final.csv:** 4,427,649 rows × 10 features
- **y_final.csv:** 4,427,649 labels (Accident_Severity)

---

## Class Distribution
| Class | Label | Count | % |
|-------|-------|-------|---|
| 1 | Fatal | 85,223 | 1.9% |
| 2 | Serious | 616,609 | 13.9% |
| 3 | Slight | 3,725,817 | 84.1% |

---

## Train / Test Split
| Set | Rows | Columns |
|-----|------|---------|
| X_train | 3,542,119 | 10 |
| X_test | 885,530 | 10 |

- Split: 80% train / 20% test, stratified by class

---

## Class Imbalance Strategy
- **Method:** `class_weight='balanced'` (no synthetic data)
- **Why not SMOTE:** Most features are label-encoded categoricals — SMOTE would interpolate between codes (e.g. Road_Type=1.7) which have no real-world meaning
- **Sample weight range:** min=0.3961 → max=17.3180
  - Fatal accidents receive ~17× more weight than Slight during training

---

## Model Results

### Logistic Regression (Baseline)
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Fatal | 0.05 | 0.69 | 0.09 | 17,045 |
| Serious | 0.15 | 0.18 | 0.16 | 123,322 |
| Slight | 0.89 | 0.59 | 0.71 | 745,163 |
| **Macro avg** | **0.36** | **0.48** | **0.32** | 885,530 |
| Weighted avg | 0.77 | 0.53 | 0.62 | 885,530 |

- Train Accuracy: 0.5309 | Test Accuracy: 0.5310 | F1-Macro: 0.3190

---

### Random Forest (200 trees, class_weight='balanced')
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Fatal | 0.08 | 0.74 | 0.15 | 17,045 |
| Serious | 0.25 | 0.39 | 0.31 | 123,322 |
| Slight | 0.91 | 0.66 | 0.77 | 745,163 |
| **Macro avg** | **0.42** | **0.60** | **0.41** | 885,530 |
| Weighted avg | 0.81 | 0.63 | 0.69 | 885,530 |

- Train Accuracy: 0.6353 | Test Accuracy: 0.6279 | F1-Macro: 0.4087

**Confusion Matrix:**
|  | Pred Fatal | Pred Serious | Pred Slight |
|--|-----------|-------------|------------|
| **Actual Fatal** | 12,646 | 2,333 | 2,066 |
| **Actual Serious** | 30,818 | 48,419 | 44,085 |
| **Actual Slight** | 110,394 | 139,786 | 494,983 |

---

### XGBoost (200 trees, sample_weight='balanced')
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Fatal | 0.07 | 0.71 | 0.12 | 17,045 |
| Serious | 0.21 | 0.30 | 0.25 | 123,322 |
| Slight | 0.90 | 0.65 | 0.75 | 745,163 |
| **Macro avg** | **0.40** | **0.55** | **0.38** | 885,530 |
| Weighted avg | 0.79 | 0.60 | 0.67 | 885,530 |

- Train Accuracy: 0.5993 | Test Accuracy: 0.5993 | F1-Macro: 0.3752

**Confusion Matrix:**
|  | Pred Fatal | Pred Serious | Pred Slight |
|--|-----------|-------------|------------|
| **Actual Fatal** | 12,033 | 2,478 | 2,534 |
| **Actual Serious** | 38,230 | 36,990 | 48,102 |
| **Actual Slight** | 129,911 | 133,608 | 481,644 |

---

## Model Comparison Summary
| Model | Train Acc | Test Acc | F1-Macro |
|-------|-----------|----------|----------|
| Logistic Regression | 0.5309 | 0.5310 | 0.3190 |
| XGBoost | 0.5993 | 0.5993 | 0.3752 |
| **Random Forest** | **0.6353** | **0.6279** | **0.4087** |

**Best model: Random Forest** (highest F1-Macro: 0.4087)

---

## Random Forest: Accuracy vs Number of Estimators
| n_estimators | Train Acc | Test Acc |
|-------------|-----------|----------|
| 10 | 0.6778 | 0.6392 |
| 50 | 0.6813 | 0.6422 |
| 100 | 0.6832 | 0.6440 |
| 150 | 0.6837 | 0.6444 |
| 200 | 0.6834 | 0.6443 |

> Performance plateaus after ~30 estimators. 200 used for final model.

---

## Output Files
| File | Description |
|------|-------------|
| `confusion_matrix.png` | Best model (RF) confusion matrix heatmap |
| `roc_curves.png` | One-vs-rest ROC curves for all 3 classes |
| `accuracy_vs_estimators.png` | RF train/test accuracy vs n_estimators |
| `classification_results.txt` | Raw results table |
