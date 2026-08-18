"""Small spectral-analysis helpers used by the public demo workflow."""

from __future__ import annotations
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def standardize_and_pca(df: pd.DataFrame, columns: list[str], n_components: int = 2):
    """Return PCA scores and fitted objects for selected spectral features."""
    x = df[columns].astype(float).to_numpy()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(x_scaled)
    score_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(n_components)], index=df.index)
    return score_df, scaler, pca


def red_edge_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["ndvi", "ndre", "gndvi", "red_edge_slope", "nir_mean", "swir_mean"]
    out = df[cols].describe().T
    out.index.name = "spectral_feature"
    return out
