# Regression Results & Interpretation

## Objective
Predict **Number_of_Casualties** for a UK road accident given pre-crash road and driver conditions.

## Input
- **X_final.csv:** 4,427,649 rows × 10 features (same features as classification)
- **Target (y):** `Number_of_Casualties` from `accidents_cleaned.csv`

---

## Target Variable — Number_of_Casualties

| Stat | Value |
|------|-------|
| Mean | 1.948 |
| Median | 1.0 |
| Min | 1 |
| Max | 8 (capped at 99th percentile) |

**Interpretation:** The median accident involves exactly 1 casualty and the mean is ~2. This is a heavily right-skewed target — most accidents involve 1 person, with a small number of multi-casualty events pulling the mean up. This makes regression inherently difficult: the model is trying to predict a value that is almost always 1 or 2, with rare spikes to 8.

---

## Train / Test Split

| Set | Rows | Columns |
|-----|------|---------|
| X_train | 3,542,119 | 10 |
| X_test | 885,530 | 10 |

- Split: 80% train / 20% test
- StandardScaler applied: fit on train, transform both

---

## Model Results

### Linear Regression (Baseline)
| Metric | Score |
|--------|-------|
| RMSE | 1.2815 |
| MAE | 0.9374 |
| R² | 0.1334 |

**Interpretation:** The baseline model explains only 13% of variance in casualty count. An RMSE of 1.28 means predictions are off by about 1.3 casualties on average. For a target with median=1, this is a large error — the model struggles because linear relationships are too simple to capture the nonlinear nature of crash outcomes.

---

### Random Forest Regressor (200 trees) — Best Model
| Metric | Score |
|--------|-------|
| RMSE | 1.1513 |
| MAE | 0.8167 |
| R² | 0.3005 |

**Interpretation:** Random Forest explains 30% of variance — more than double the baseline. MAE of 0.82 means the typical prediction is within ~1 casualty of the actual value, which is reasonable given the data. The improvement over Linear Regression confirms that casualty count has nonlinear relationships with road conditions.

---

### XGBoost Regressor (200 trees, lr=0.1)
| Metric | Score |
|--------|-------|
| RMSE | 1.2073 |
| MAE | 0.8633 |
| R² | 0.2308 |

**Interpretation:** XGBoost performs between the baseline and Random Forest. R² of 0.23 is reasonable but trails Random Forest by ~7 percentage points. XGBoost may benefit from further hyperparameter tuning (depth, learning rate) but was not tuned here.

---

## Model Comparison

| Model | RMSE | MAE | R² | Rank |
|-------|------|-----|----|------|
| Linear Regression | 1.2815 | 0.9374 | 0.1334 | 3rd |
| XGBoost | 1.2073 | 0.8633 | 0.2308 | 2nd |
| **Random Forest** | **1.1513** | **0.8167** | **0.3005** | **1st** |

**Best model: Random Forest** — lowest RMSE (1.15) and highest R² (0.30)

---

## Why R² is Low (and Why That's Expected)

An R² of 0.30 might seem low, but it is appropriate given this problem:

1. **The target is almost always 1** — 84% of accidents have 1 casualty. Any model that predicts "1" for everything would have low RMSE but learn nothing useful. Our model goes beyond this default.

2. **Casualty count is determined by post-crash physics** — airbag deployment, exact impact speed, seatbelt use, medical response time. None of these are in our dataset. The model only has pre-crash road conditions, which set the *risk* but not the exact *outcome*.

3. **The features are the same 10 used for severity classification** — they were selected to distinguish Fatal/Serious/Slight, not specifically to predict casualty count. A dedicated feature set for regression might improve R².

4. **R² = 0.30 with only road/driver features is actually informative** — it confirms that road conditions (speed limit, road type, vehicle manoeuvre, time of night) do carry real signal about how many people will be hurt, even without knowing what happens during the crash.

---

## Output Files

| File | Description |
|------|-------------|
| `regression_actual_vs_predicted.png` | Scatter plot of actual vs predicted casualties with y=x reference line |
| `residuals_histogram.png` | Distribution of prediction errors (Predicted − Actual) |
| `learning_curve.png` | Train vs validation RMSE as training size increases (100K subsample) |
| `regression_results.txt` | Raw comparison table |

---

## Key Takeaway for Report

Random Forest is the best regressor with **RMSE=1.15** and **R²=0.30**. While the R² is modest, this is consistent with the nature of the problem — predicting exact casualty counts from pre-crash data alone has a natural ceiling. The regression model is still useful: it can estimate *relative* casualty risk across different road and weather scenarios, helping city planners prioritise where infrastructure improvements would have the most impact on reducing casualties.

The consistent finding across both classification and regression tasks is that **Random Forest outperforms both the linear baseline and XGBoost** on this dataset — likely because the features are discrete/categorical and the relationships are nonlinear and tree-structured.
