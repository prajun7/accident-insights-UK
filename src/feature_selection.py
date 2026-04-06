import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1 — LOAD
    # ------------------------------------------------------------------
    print("\n--- STEP 1: Load ---")
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'accidents_cleaned.csv'), low_memory=False)
    print(f"Loaded shape: {df.shape}")

    # ------------------------------------------------------------------
    # STEP 2 — SELECT COLUMNS
    # ------------------------------------------------------------------
    print("\n--- STEP 2: Select Columns ---")

    # Save lat/lon before dropping
    lat_lon = df[['Latitude', 'Longitude']].copy()
    lat_lon.to_csv(os.path.join(OUTPUT_DIR, 'lat_lon.csv'), index=False)
    print(f"Saved lat_lon.csv — shape: {lat_lon.shape}")

    keep_cols = [
        'Accident_Severity', 'Speed_limit', 'Road_Type', 'Light_Conditions',
        'Weather_Conditions', 'Road_Surface_Conditions', 'Urban_or_Rural_Area',
        'Junction_Detail', 'Junction_Control', 'Number_of_Vehicles',
        'Number_of_Casualties', 'Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'IsNight'
    ]

    # Only keep columns that exist in the dataframe
    keep_cols = [c for c in keep_cols if c in df.columns]
    missing = [c for c in [
        'Accident_Severity', 'Speed_limit', 'Road_Type', 'Light_Conditions',
        'Weather_Conditions', 'Road_Surface_Conditions', 'Urban_or_Rural_Area',
        'Junction_Detail', 'Junction_Control', 'Number_of_Vehicles',
        'Number_of_Casualties', 'Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'IsNight'
    ] if c not in df.columns]
    if missing:
        print(f"Warning — columns not found (skipped): {missing}")

    df = df[keep_cols].copy()
    print(f"Shape after column selection: {df.shape}")
    print(f"Columns kept: {keep_cols}")

    # ------------------------------------------------------------------
    # STEP 3 — LABEL ENCODING
    # ------------------------------------------------------------------
    print("\n--- STEP 3: Label Encoding ---")
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == object:
            print(f"  Encoding '{col}': {df[col].nunique()} unique values → integers")
            df[col] = le.fit_transform(df[col].astype(str))

    # ------------------------------------------------------------------
    # STEP 4 — CORRELATION MATRIX
    # ------------------------------------------------------------------
    print("\n--- STEP 4: Correlation Matrix ---")
    corr = df.corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'))
    plt.close()
    print("Saved correlation_matrix.png")

    # Drop one column from highly correlated pairs (|corr| > 0.85)
    upper = corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_corr = [col for col in upper.columns if any(upper[col] > 0.85)]
    if drop_corr:
        print(f"Dropping highly correlated columns (|corr| > 0.85): {drop_corr}")
        df.drop(columns=drop_corr, inplace=True)
    else:
        print("No columns dropped for high correlation.")

    # ------------------------------------------------------------------
    # STEP 5 — SEPARATE TARGET
    # ------------------------------------------------------------------
    print("\n--- STEP 5: Separate Target ---")
    X = df.drop(columns=['Accident_Severity'])
    y = df['Accident_Severity']
    print(f"X shape: {X.shape} | y shape: {y.shape}")

    # ------------------------------------------------------------------
    # STEP 6 — FEATURE IMPORTANCE (Random Forest)
    # ------------------------------------------------------------------
    print("\n--- STEP 6: Feature Importance (Random Forest) ---")
    print("Training Random Forest — this may take a minute...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top10 = importances.head(10)
    print(f"\nTop 10 features:\n{top10}")

    plt.figure(figsize=(10, 6))
    top10.sort_values().plot(kind='barh', color='steelblue')
    plt.title('Top 10 Feature Importances — Random Forest')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))
    plt.close()
    print("Saved feature_importance.png")

    X = X[top10.index.tolist()]
    print(f"X reduced to top 10 features: {X.shape}")

    # ------------------------------------------------------------------
    # STEP 7 — PCA (for clustering)
    # ------------------------------------------------------------------
    print("\n--- STEP 7: PCA ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Explained variance ratio: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df.to_csv(os.path.join(OUTPUT_DIR, 'X_pca.csv'), index=False)
    print("Saved X_pca.csv")

    plt.figure(figsize=(10, 7))
    severity_labels = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
    colors = {1: 'red', 2: 'orange', 3: 'steelblue'}
    for sev in sorted(y.unique()):
        mask = y.values == sev
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    label=severity_labels.get(sev, str(sev)),
                    color=colors.get(sev, 'gray'),
                    alpha=0.3, s=2)
    plt.title('PCA 2D Projection — Colored by Accident Severity')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(markerscale=5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pca_scatter.png'))
    plt.close()
    print("Saved pca_scatter.png")

    # ------------------------------------------------------------------
    # STEP 8 — SAVE FINAL FEATURES
    # ------------------------------------------------------------------
    print("\n--- STEP 8: Save Final Features ---")
    X.to_csv(os.path.join(OUTPUT_DIR, 'X_final.csv'), index=False)
    y.to_csv(os.path.join(OUTPUT_DIR, 'y_final.csv'), index=False)
    print(f"Saved X_final.csv — shape: {X.shape}")
    print(f"Saved y_final.csv — shape: {y.shape}")
    print(f"\nFinal selected features: {X.columns.tolist()}")


if __name__ == "__main__":
    run()
