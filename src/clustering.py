import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1 — LOAD AND NORMALIZE X_PCA
    # ------------------------------------------------------------------
    print("\n--- STEP 1: Load and Normalize ---")
    pca_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'X_pca.csv'))
    print(f"X_pca shape: {pca_df.shape}")

    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(pca_df)
    print(f"Normalized to [0,1] range — PC1: [{X_norm[:,0].min():.3f}, {X_norm[:,0].max():.3f}]  "
          f"PC2: [{X_norm[:,1].min():.3f}, {X_norm[:,1].max():.3f}]")

    # ------------------------------------------------------------------
    # STEP 2 — ELBOW METHOD
    # ------------------------------------------------------------------
    print("\n--- STEP 2: Elbow Method (k=2 to 12) ---")
    inertias = []
    k_range  = range(2, 13)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_norm)
        inertias.append(km.inertia_)
        print(f"  k={k:>2}: inertia={km.inertia_:,.1f}")

    plt.figure(figsize=(10, 6))
    plt.plot(list(k_range), inertias, marker='o', linewidth=2, color='steelblue')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method — Optimal Number of Clusters')
    plt.xticks(list(k_range))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'elbow_plot.png'))
    plt.close()
    print("Saved elbow_plot.png")

    # Choose best_k — look for the elbow (biggest drop slowdown)
    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    best_k = int(k_range[int(np.argmax(diffs2)) + 1])
    if best_k < 3:
        best_k = 4   # floor at 4 for interpretability
    print(f"Selected best_k={best_k} (elbow point where inertia drop slows most)")

    # ------------------------------------------------------------------
    # STEP 3 — TRAIN K-MEANS
    # ------------------------------------------------------------------
    print(f"\n--- STEP 3: K-Means (k={best_k}) ---")
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    km_labels = km_final.fit_predict(X_norm)

    sil_km = silhouette_score(X_norm, km_labels, sample_size=50000, random_state=42)
    db_km  = davies_bouldin_score(X_norm, km_labels)
    print(f"Silhouette Score     : {sil_km:.4f}  (higher is better, max=1)")
    print(f"Davies-Bouldin Index : {db_km:.4f}  (lower is better, min=0)")

    # ------------------------------------------------------------------
    # STEP 4 — PLOT K-MEANS CLUSTERS
    # ------------------------------------------------------------------
    print("\n--- STEP 4: K-Means Cluster Plot ---")
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_norm[:, 0], X_norm[:, 1],
                          c=km_labels, cmap='tab10', alpha=0.3, s=2)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'K-Means Clustering of Accident Feature Space (k={best_k})')
    plt.xlabel('PC1 (normalised)')
    plt.ylabel('PC2 (normalised)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'kmeans_clusters.png'))
    plt.close()
    print("Saved kmeans_clusters.png")

    # ------------------------------------------------------------------
    # STEP 5 — CLUSTER PROFILES
    # ------------------------------------------------------------------
    print("\n--- STEP 5: Cluster Profiles ---")
    accidents = pd.read_csv(os.path.join(OUTPUT_DIR, 'accidents_cleaned.csv'),
                            low_memory=False)

    # Align rows — X_pca and accidents_cleaned have the same row order
    accidents = accidents.reset_index(drop=True).iloc[:len(km_labels)]
    accidents['KMeans_Cluster'] = km_labels

    profile_cols = ['Speed_limit', 'Hour', 'IsNight', 'Urban_or_Rural_Area',
                    'Accident_Severity', 'Number_of_Casualties']
    profile_cols = [c for c in profile_cols if c in accidents.columns]

    severity_map = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
    print("\nCluster profiles (mean values):")
    profile = accidents.groupby('KMeans_Cluster')[profile_cols].mean().round(3)
    print(profile.to_string())

    # Save cluster profiles as CSV
    profile.to_csv(os.path.join(OUTPUT_DIR, 'kmeans_cluster_profiles.csv'))
    print("Saved kmeans_cluster_profiles.csv")

    # ------------------------------------------------------------------
    # STEP 6 — DBSCAN (on sample — DBSCAN does not scale to 4.4M rows)
    # ------------------------------------------------------------------
    # DBSCAN computes pairwise distances which is O(n²) memory.
    # Running on 4.4M points exhausts RAM and gets killed by the OS.
    # Standard practice for large datasets: run DBSCAN on a representative
    # sample. 200K points preserves density patterns while staying in memory.
    # ------------------------------------------------------------------
    DBSCAN_SAMPLE = 200_000
    print(f"\n--- STEP 6: DBSCAN on {DBSCAN_SAMPLE:,}-row sample (eps=0.05, min_samples=10) ---")
    print(f"Note: DBSCAN sampled from {len(X_norm):,} total rows to avoid OOM kill.")

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_norm), size=DBSCAN_SAMPLE, replace=False)
    sample_idx.sort()
    X_db_sample = X_norm[sample_idx]

    db = DBSCAN(eps=0.05, min_samples=10)
    db_labels = db.fit_predict(X_db_sample)

    n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise       = int((db_labels == -1).sum())
    pct_noise     = n_noise / len(db_labels) * 100

    print(f"Clusters found  : {n_clusters_db}")
    print(f"Noise points    : {n_noise:,}  ({pct_noise:.1f}%)")

    # Silhouette only on non-noise points
    mask_valid = db_labels != -1
    if mask_valid.sum() > 1 and n_clusters_db > 1:
        sil_db = silhouette_score(X_db_sample[mask_valid], db_labels[mask_valid],
                                  sample_size=min(50000, mask_valid.sum()), random_state=42)
        db_db  = davies_bouldin_score(X_db_sample[mask_valid], db_labels[mask_valid])
        print(f"Silhouette Score (non-noise)     : {sil_db:.4f}")
        print(f"Davies-Bouldin Index (non-noise) : {db_db:.4f}")
    else:
        sil_db = float('nan')
        db_db  = float('nan')
        print("Not enough clusters for silhouette score.")

    # ------------------------------------------------------------------
    # STEP 7 — PLOT DBSCAN CLUSTERS
    # ------------------------------------------------------------------
    print("\n--- STEP 7: DBSCAN Cluster Plot ---")
    plt.figure(figsize=(10, 7))
    unique_labels = sorted(set(db_labels))
    cmap = plt.cm.get_cmap('tab20', len(unique_labels))

    for i, label in enumerate(unique_labels):
        mask = db_labels == label
        color = 'lightgray' if label == -1 else cmap(i)
        lname = 'Noise' if label == -1 else f'Cluster {label}'
        plt.scatter(X_db_sample[mask, 0], X_db_sample[mask, 1],
                    c=[color], alpha=0.3, s=2, label=lname)

    plt.title('DBSCAN Clustering — Accident Pattern Hotspots')
    plt.xlabel('PC1 (normalised)')
    plt.ylabel('PC2 (normalised)')
    if n_clusters_db <= 10:
        plt.legend(markerscale=5, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dbscan_clusters.png'))
    plt.close()
    print("Saved dbscan_clusters.png")

    # ------------------------------------------------------------------
    # STEP 8 — GEOGRAPHIC HOTSPOT MAP
    # ------------------------------------------------------------------
    print("\n--- STEP 8: Geographic Hotspot Map ---")
    lat_lon = pd.read_csv(os.path.join(OUTPUT_DIR, 'lat_lon.csv'))
    lat_lon = lat_lon.reset_index(drop=True).iloc[sample_idx].reset_index(drop=True)
    lat_lon['DBSCAN_Cluster'] = db_labels

    # Remove noise points and rows with missing coordinates
    geo = lat_lon[(lat_lon['DBSCAN_Cluster'] != -1) &
                  lat_lon['Latitude'].notna() &
                  lat_lon['Longitude'].notna()].copy()

    print(f"Plotting {len(geo):,} non-noise points across {n_clusters_db} clusters")

    plt.figure(figsize=(12, 8))
    unique_clusters = sorted(geo['DBSCAN_Cluster'].unique())
    cmap_geo = plt.cm.get_cmap('tab20', len(unique_clusters))

    for i, cluster in enumerate(unique_clusters):
        mask = geo['DBSCAN_Cluster'] == cluster
        plt.scatter(geo.loc[mask, 'Longitude'], geo.loc[mask, 'Latitude'],
                    c=[cmap_geo(i)], alpha=0.3, s=5,
                    label=f'Cluster {cluster}')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Geographic Accident Hotspots — DBSCAN Clusters (UK)')
    if n_clusters_db <= 15:
        plt.legend(markerscale=3, loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'geographic_hotspots.png'))
    plt.close()
    print("Saved geographic_hotspots.png")

    # ------------------------------------------------------------------
    # STEP 9 — COMPARISON TABLE
    # ------------------------------------------------------------------
    print("\n--- STEP 9: Comparison Table ---")
    summary = (
        f"UK TRAFFIC ACCIDENT — CLUSTERING RESULTS\n"
        f"{'='*55}\n\n"
        f"{'Method':<12} {'Silhouette':>12} {'Davies-Bouldin':>16} {'Clusters':>10}\n"
        f"{'-'*55}\n"
        f"{'K-Means':<12} {sil_km:>12.4f} {db_km:>16.4f} {best_k:>10}\n"
        f"{'DBSCAN':<12} {sil_db if not np.isnan(sil_db) else 'N/A':>12} "
        f"{db_db if not np.isnan(db_db) else 'N/A':>16} {n_clusters_db:>10}\n\n"
        f"DBSCAN noise points: {n_noise:,} ({pct_noise:.1f}%)\n"
        f"K-Means best_k selected: {best_k}\n"
    )
    print(summary)

    results_path = os.path.join(OUTPUT_DIR, 'clustering_results.txt')
    with open(results_path, 'w') as f:
        f.write(summary)
    print("Saved clustering_results.txt")

    print("\n" + "=" * 55)
    print("CLUSTERING COMPLETE")
    print("=" * 55)
    print(f"  K-Means clusters  : {best_k}")
    print(f"  K-Means Silhouette: {sil_km:.4f}")
    print(f"  DBSCAN clusters   : {n_clusters_db}  (noise: {pct_noise:.1f}%)")
    print("=" * 55)


if __name__ == "__main__":
    run()
