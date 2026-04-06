# Preprocessing Results

## Dataset Files (Raw)
| File | Rows Loaded | Columns |
|------|-------------|---------|
| `Accidents0515.csv` | 1,780,653 | 32 |
| `Casualties0515.csv` | 2,216,720 | 15 |
| `Vehicles0515.csv` | 3,004,425 | 22 |

> Note: Casualties and Vehicles row counts are lower than raw file line counts due to malformed lines being skipped (`on_bad_lines='skip'`).

---

## Step 2 — Merged Shape
- **After merge:** 4,427,649 rows × 67 columns

---

## Step 3 — Dropped Columns (>40% null)
- None dropped

---

## Step 5 — Null Imputation
| Column | Nulls Before |
|--------|-------------|
| Pedestrian_Road_Maintenance_Worker | 2,919,786 |
| 2nd_Road_Class | 1,843,503 |
| Junction_Control | 1,590,993 |
| Driver_IMD_Decile | 1,302,734 |
| Age_of_Vehicle | 1,295,544 |
| Propulsion_Code | 1,142,210 |
| Engine_Capacity_(CC) | 1,168,745 |
| Driver_Home_Area_Type | 924,988 |
| Casualty_Home_Area_Type | 741,592 |
| Age_of_Driver | 545,720 |
| *(+ 44 other columns with smaller null counts)* | — |

**Null count after imputation: 0** ✓

---

## Step 7 — Feature Engineering (5 new columns added)
| Column | Description |
|--------|-------------|
| `Month` | Extracted from Date (1–12) |
| `DayOfWeek` | 0 = Monday, 6 = Sunday |
| `IsWeekend` | 1 if DayOfWeek ≥ 5 |
| `Hour` | Extracted from Time (0–23) |
| `IsNight` | 1 if Hour < 6 or Hour ≥ 20 |

---

## Step 8 — Outlier Caps (99th Percentile)
| Column | Cap Value |
|--------|-----------|
| `Number_of_Casualties` | 8.0 |
| `Speed_limit` | 70.0 |

---

## Final Cleaned Dataset
- **Shape:** 4,427,649 rows × 72 columns
- **Saved to:** `output/accidents_cleaned.csv`

---

## Accident_Severity Distribution
| Code | Label | Count | Percentage |
|------|-------|-------|------------|
| 1 | Fatal | 85,223 | 1.9% |
| 2 | Serious | 616,609 | 13.9% |
| 3 | Slight | 3,725,817 | 84.1% |

---

## Date Range
**2005-01-01 → 2015-12-31**
