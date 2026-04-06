# Feature Selection Results

## Input
- **Source:** `output/accidents_cleaned.csv`
- **Loaded shape:** 4,427,649 rows × 72 columns

---

## Step 2 — Column Selection
| Decision | Columns |
|----------|---------|
| **Kept** | Accident_Severity, Speed_limit, Road_Type, Light_Conditions, Weather_Conditions, Road_Surface_Conditions, Urban_or_Rural_Area, Junction_Detail, Junction_Control, Number_of_Vehicles, Hour, IsNight |
| **Removed — frequency predictors** | Month, DayOfWeek, IsWeekend |
| **Removed — data leakage** | Number_of_Casualties |

> `Number_of_Casualties` removed because casualty count is only known *after* the accident — same time as severity is recorded. Including it would inflate model accuracy artificially.
> `Month`, `DayOfWeek`, `IsWeekend` removed because they predict *when* accidents happen (frequency), not *how bad* they are (severity).

- **Shape after selection:** 4,427,649 rows × 12 columns

---

## Step 3 — Label Encoding
No object-type columns found — all selected columns were already numeric.

---

## Step 4 — Correlation Matrix
- **Saved:** `output/correlation_matrix.png`
- **Columns dropped (|corr| > 0.85):** None

---

## Step 5 — Class Distribution
| Severity | Label | Count | % |
|----------|-------|-------|---|
| 1 | Fatal | 85,223 | 1.9% |
| 2 | Serious | 616,609 | 13.9% |
| 3 | Slight | 3,725,817 | 84.1% |

- **X shape:** (4,427,649 × 11) | **y shape:** (4,427,649,)

---

## Step 6 — LDA (Linear Discriminant Analysis)

**Why LDA over PCA or Random Forest importance?**
- LDA uses class labels (Fatal/Serious/Slight) to find directions that **maximise separation between severity classes**
- PCA ignores class labels — it maximises general variance, not class separation
- With 4.4M labelled rows and 3 classes, LDA is highly stable and interpretable

**LDA Explained Variance:**
| Component | Variance Explained |
|-----------|-------------------|
| LD1 | 97.2% |
| LD2 | 2.8% |
| **Total** | **100.0%** |

**LDA Feature Coefficients (sorted by |LD1|):**
| Feature | LD1 Coefficient | LD2 Coefficient |
|---------|----------------|----------------|
| Speed_limit | -0.4535 | 0.7255 |
| Number_of_Vehicles | -0.4144 | -0.3123 |
| Road_Type | -0.3589 | -0.5448 |
| IsNight | -0.3436 | 0.1471 |
| Urban_or_Rural_Area | -0.3279 | -0.4057 |
| Junction_Detail | 0.1725 | -0.0199 |
| Light_Conditions | -0.1695 | -0.1441 |
| Road_Surface_Conditions | 0.1177 | -0.0179 |
| Hour | 0.0689 | -0.0706 |
| Weather_Conditions | 0.0663 | 0.4153 |
| Junction_Control | -0.0171 | -0.2339 |

**Top 10 features selected (by |LD1| coefficient):**
1. Speed_limit
2. Number_of_Vehicles
3. Road_Type
4. IsNight
5. Urban_or_Rural_Area
6. Junction_Detail
7. Light_Conditions
8. Road_Surface_Conditions
9. Hour
10. Weather_Conditions

> `Junction_Control` was the lowest-ranked feature and was dropped.

**Saved:** `output/lda_feature_coefficients.png`, `output/lda_scatter.png`, `output/X_lda.csv`

---

## Step 7 — PCA (for clustering visualisation only)

> PCA is used here **only** to produce a 2D input for K-Means and DBSCAN clustering. It is not used for feature selection — LDA handles that.

| Component | Variance Explained |
|-----------|-------------------|
| PC1 | 20.1% |
| PC2 | 17.3% |
| **Total** | **37.4%** |

**Saved:** `output/X_pca.csv`, `output/pca_scatter.png`

---

## Step 8 — Final Outputs

| File | Shape | Description |
|------|-------|-------------|
| `X_final.csv` | 4,427,649 × 10 | Final feature matrix (top 10 by LDA) |
| `y_final.csv` | 4,427,649 × 1 | Target: Accident_Severity |
| `X_pca.csv` | 4,427,649 × 2 | 2D PCA (for clustering) |
| `X_lda.csv` | 4,427,649 × 2 | 2D LDA projection |
| `lat_lon.csv` | 4,427,649 × 2 | Latitude & Longitude (for geographic map) |

---

## Summary

| Item | Detail |
|------|--------|
| Method | LDA (Linear Discriminant Analysis) |
| LDA variance explained | 100% (LD1=97.2%, LD2=2.8%) |
| PCA variance explained | 37.4% (clustering only) |
| Features removed | Month, DayOfWeek, IsWeekend (frequency); Number_of_Casualties (leakage) |
| Final feature count | 10 |
| Final features | Speed_limit, Number_of_Vehicles, Road_Type, IsNight, Urban_or_Rural_Area, Junction_Detail, Light_Conditions, Road_Surface_Conditions, Hour, Weather_Conditions |
