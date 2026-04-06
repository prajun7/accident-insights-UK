# Feature Selection Results — V2 (22 input columns)

## Input
- **Source:** `output/accidents_cleaned.csv`
- **Loaded shape:** 4,427,649 rows × 72 columns

---

## Step 2 — Column Selection
**22 columns selected (21 features + target)**

| Category | Columns Kept |
|----------|-------------|
| Road conditions | Speed_limit, Road_Type, 1st_Road_Class, Light_Conditions, Weather_Conditions, Road_Surface_Conditions, Special_Conditions_at_Site, Carriageway_Hazards, Urban_or_Rural_Area, Junction_Detail, Junction_Control, Pedestrian_Crossing-Physical_Facilities |
| Accident context | Number_of_Vehicles, Hour, IsNight |
| Vehicle features | Vehicle_Type, Vehicle_Manoeuvre, Towing_and_Articulation |
| Driver features | Age_Band_of_Driver, Sex_of_Driver, Journey_Purpose_of_Driver |

**Removed — frequency predictors:** Month, DayOfWeek, IsWeekend

**Removed — data leakage (known only after crash):** Number_of_Casualties, Casualty_Severity, Skidding_and_Overturning, Hit_Object_in_Carriageway, Vehicle_Leaving_Carriageway, 1st_Point_of_Impact

- **Shape after selection:** 4,427,649 rows × 22 columns

---

## Step 3 — Label Encoding
No object-type columns — all selected columns were already numeric.

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

- **X shape:** (4,427,649 × 21) | **y shape:** (4,427,649,)

---

## Step 6 — LDA (Linear Discriminant Analysis)

| Component | Variance Explained |
|-----------|-------------------|
| LD1 | 95.8% |
| LD2 | 4.2% |
| **Total** | **100.0%** |

**LDA Feature Coefficients (all 21, sorted by |LD1|):**
| Rank | Feature | LD1 | LD2 |
|------|---------|-----|-----|
| 1 | Speed_limit | -0.3945 | 0.3829 |
| 2 | Number_of_Vehicles | -0.3921 | -0.1976 |
| 3 | Vehicle_Manoeuvre | -0.3649 | -0.3291 |
| 4 | Road_Type | -0.3126 | -0.2233 |
| 5 | IsNight | -0.2941 | 0.1784 |
| 6 | Urban_or_Rural_Area | -0.2809 | -0.2357 |
| 7 | Sex_of_Driver | 0.1978 | 0.0759 |
| 8 | Junction_Detail | 0.1618 | -0.0425 |
| 9 | Age_Band_of_Driver | -0.1520 | 0.0062 |
| 10 | Light_Conditions | -0.1470 | -0.0734 |
| 11 | Road_Surface_Conditions | 0.1045 | 0.0014 |
| 12 | Journey_Purpose_of_Driver | -0.0718 | 0.0708 |
| 13 | Pedestrian_Crossing-Physical_Facilities | -0.0665 | -0.1142 |
| 14 | Hour | 0.0628 | -0.0526 |
| 15 | Weather_Conditions | 0.0619 | 0.2902 |
| 16 | Towing_and_Articulation | -0.0612 | 0.0962 |
| 17 | Special_Conditions_at_Site | 0.0599 | -0.1102 |
| 18 | 1st_Road_Class | -0.0394 | -0.4883 |
| 19 | Junction_Control | -0.0274 | -0.1456 |
| 20 | Carriageway_Hazards | 0.0253 | 0.1169 |
| 21 | Vehicle_Type | 0.0028 | 0.3264 |

**Top 10 features selected (by |LD1| coefficient):**
1. Speed_limit
2. Number_of_Vehicles
3. Vehicle_Manoeuvre *(new — V2)*
4. Road_Type
5. IsNight
6. Urban_or_Rural_Area
7. Sex_of_Driver *(new — V2)*
8. Junction_Detail
9. Age_Band_of_Driver *(new — V2)*
10. Light_Conditions

> 3 of the 10 final features are new additions from V2 — Vehicle_Manoeuvre, Sex_of_Driver, Age_Band_of_Driver replaced Road_Surface_Conditions, Hour, and Weather_Conditions.

**Saved:** `output/lda_feature_coefficients.png`, `output/lda_scatter.png`, `output/X_lda.csv`

---

## Step 7 — PCA (for clustering only)

| Component | Variance Explained |
|-----------|-------------------|
| PC1 | 19.8% |
| PC2 | 16.5% |
| **Total** | **36.3%** |

**Saved:** `output/X_pca.csv`, `output/pca_scatter.png`

---

## Step 8 — Final Outputs

| File | Shape | Description |
|------|-------|-------------|
| `X_final.csv` | 4,427,649 × 10 | Final feature matrix (top 10 by LDA) |
| `y_final.csv` | 4,427,649 × 1 | Target: Accident_Severity |
| `X_pca.csv` | 4,427,649 × 2 | 2D PCA (for clustering) |
| `X_lda.csv` | 4,427,649 × 2 | 2D LDA projection |
| `lat_lon.csv` | 4,427,649 × 2 | Latitude & Longitude (geographic map) |

---

## V1 vs V2 Comparison

| | V1 | V2 |
|-|----|----|
| Input features | 11 | 21 |
| Driver features | None | Age_Band_of_Driver, Sex_of_Driver, Journey_Purpose_of_Driver |
| Vehicle features | None | Vehicle_Type, Vehicle_Manoeuvre, Towing_and_Articulation |
| LDA LD1 variance | 97.2% | 95.8% |
| Final features (top 10) | Speed_limit, Number_of_Vehicles, Road_Type, IsNight, Urban_or_Rural_Area, Junction_Detail, Light_Conditions, Road_Surface_Conditions, Hour, Weather_Conditions | Speed_limit, Number_of_Vehicles, **Vehicle_Manoeuvre**, Road_Type, IsNight, Urban_or_Rural_Area, **Sex_of_Driver**, Junction_Detail, **Age_Band_of_Driver**, Light_Conditions |

---

## Summary

| Item | Detail |
|------|--------|
| Method | LDA (Linear Discriminant Analysis) |
| LDA variance explained | 100% (LD1=95.8%, LD2=4.2%) |
| PCA variance explained | 36.3% (clustering only) |
| Final feature count | 10 |
| New features that made top 10 | Vehicle_Manoeuvre, Sex_of_Driver, Age_Band_of_Driver |
| Features dropped from top 10 vs V1 | Road_Surface_Conditions, Hour, Weather_Conditions |
