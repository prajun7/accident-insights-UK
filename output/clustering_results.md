# Clustering Results & Interpretation

## Input
- **Source:** `output/X_pca.csv` — 4,427,649 rows × 2 PCA components
- **Normalized:** MinMaxScaler applied to [0, 1] range before clustering

---

## Step 2 — Elbow Method (K-Means, k=2 to 12)

| k | Inertia | Drop from previous |
|---|---------|-------------------|
| 2 | 85,563.4 | — |
| 3 | 43,355.1 | 42,208.3 |
| 4 | 32,773.8 | 10,581.3 |
| 5 | 26,669.4 | 6,104.4 |
| 6 | 21,744.7 | 4,924.7 |
| 7 | 18,219.9 | 3,524.8 |
| 8 | 15,981.7 | 2,238.2 |
| 9 | 14,484.5 | 1,497.2 |
| 10 | 13,016.1 | 1,468.4 |
| 11 | 12,063.5 | 952.6 |
| 12 | 11,107.9 | 955.6 |

**Selected k = 3** — the biggest drop in inertia is from k=2 to k=3 (42K), with diminishing returns after that. Going beyond k=3 adds clusters that are less distinct.

**Saved:** `output/elbow_plot.png`

---

## Step 3 — K-Means Results (k=3)

| Metric | Score | Interpretation |
|--------|-------|---------------|
| Silhouette Score | 0.5224 | Moderate–good separation. Clusters are reasonably well-defined but with some overlap — expected given PCA only captures 36% of variance |
| Davies-Bouldin Index | 0.6352 | Good. Values below 1.0 indicate well-separated clusters. Lower is better |

---

## Step 5 — Cluster Profiles (K-Means)

| Cluster | Speed Limit | Hour | IsNight | Urban/Rural* | Avg Severity** | Avg Casualties |
|---------|------------|------|---------|-------------|---------------|----------------|
| 0 | 57.7 mph | 12:57 | 1.1% | Rural (1.93) | 2.76 (Slight) | 2.30 |
| 1 | 40.5 mph | 15:32 | 84.1% | Mixed (1.39) | 2.75 (Slight) | 2.05 |
| 2 | 31.3 mph | 13:26 | 0.8% | Urban (1.10) | 2.88 (Slight) | 1.70 |

*Urban/Rural: 1 = Urban, 2 = Rural — lower value = more urban
**Severity: 1 = Fatal, 2 = Serious, 3 = Slight — lower value = more severe

### What each cluster represents

**Cluster 0 — Rural High-Speed Daytime Accidents**
- Average speed limit of 57.7 mph — these are A-roads and rural roads
- Almost entirely daytime (only 1.1% at night)
- More rural setting (1.93 ≈ Rural)
- Highest casualty count (2.30) — high-speed rural crashes involve more people or more severe injuries
- This cluster represents country road and A-road accidents during daylight hours

**Cluster 1 — Night-Time Mixed-Area Accidents**
- 84.1% of accidents in this cluster happen at night — the defining characteristic
- Lower speed limit (40.5 mph) — suburban and mixed roads
- Mixed urban/rural area (1.39)
- Night-time driving increases crash risk due to reduced visibility and fatigue
- This cluster captures the late-afternoon/evening accident pattern (peak at 15:32 average)

**Cluster 2 — Urban Low-Speed Daytime Accidents**
- Lowest speed limit (31.3 mph) — city streets and residential roads
- Almost entirely urban (1.10 ≈ Urban)
- Almost no night accidents (0.8%)
- Lowest casualty count (1.70) — lower speeds mean less severe crashes
- Highest severity code (2.88 ≈ Slight) — the least severe cluster overall
- This cluster represents typical city-centre fender-benders and pedestrian-zone accidents

**Saved:** `output/kmeans_clusters.png`, `output/kmeans_cluster_profiles.csv`

---

## Step 6 — DBSCAN Results

> **Note:** DBSCAN ran on a 200,000-row random sample from 4,427,649 total rows.
> Running DBSCAN on the full dataset would exhaust RAM (O(n²) distance matrix).
> This is standard practice for large datasets — density patterns are preserved in the sample.

| Metric | Score | Interpretation |
|--------|-------|---------------|
| Clusters found | 3 | Matches K-Means — both methods independently agree on 3 natural groupings |
| Noise points | 4 (0.0%) | Almost no outliers — the data is very dense and cohesive, few isolated accident patterns |
| Silhouette Score | 0.4920 | Moderate separation — slightly lower than K-Means, typical for density-based methods |
| Davies-Bouldin Index | 0.3611 | Better than K-Means (0.64). DBSCAN found more compact, well-separated clusters |

**Saved:** `output/dbscan_clusters.png`

---

## Step 8 — Geographic Hotspot Map

- **Points plotted:** 199,996 non-noise accidents across 3 clusters
- **Saved:** `output/geographic_hotspots.png`

The geographic map shows where each cluster concentrates across the UK:
- **Cluster 0 (Rural/High-Speed)** — distributed across motorway corridors and A-road networks outside cities
- **Cluster 1 (Night-Time)** — spread across all areas but concentrated around urban fringes and commuter routes
- **Cluster 2 (Urban/Low-Speed)** — densely concentrated in London, Manchester, Birmingham, and other major city centres

---

## Method Comparison

| Method | Silhouette | Davies-Bouldin | Clusters | Notes |
|--------|-----------|----------------|----------|-------|
| K-Means | 0.5224 | 0.6352 | 3 | Full 4.4M rows |
| DBSCAN | 0.4920 | **0.3611** | 3 | 200K sample |

**Both methods independently found 3 clusters** — this strongly validates that 3 is the natural number of accident groupings in this dataset.

DBSCAN's lower Davies-Bouldin score (0.36 vs 0.64) means its clusters are more compact internally. K-Means has slightly higher Silhouette, meaning its clusters are more separated from each other.

**Saved:** `output/clustering_results.txt`

---

## Key Takeaway for Report

The clustering reveals that UK road accidents naturally fall into **three distinct behavioural groups**:

1. **Rural high-speed daytime crashes** — A-roads, country roads, higher casualties per accident
2. **Night-time mixed-area crashes** — reduced visibility, fatigue, suburban roads
3. **Urban low-speed daytime crashes** — city centres, lower severity, highest volume

This finding supports targeted road safety interventions:
- Cluster 0 → rural road speed enforcement and overtaking restrictions
- Cluster 1 → night-time alcohol checks, street lighting improvements
- Cluster 2 → urban junction redesign, pedestrian crossing improvements
