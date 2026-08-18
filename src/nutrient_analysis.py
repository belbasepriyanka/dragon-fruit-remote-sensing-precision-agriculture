"""Nutrient summaries for the demonstration dataset."""

from __future__ import annotations
import pandas as pd


def treatment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize crop response by environment and treatment."""
    group_cols = ["environment", "treatment_t_acre"]
    metrics = ["tissue_n_pct", "tissue_p_pct", "tissue_k_pct", "ndvi", "ndre", "flower_count", "fruit_count", "stress_flag"]
    return df.groupby(group_cols, as_index=False)[metrics].mean(numeric_only=True).sort_values(group_cols)


def late_season_k_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Transparent summary useful for checking late-season K patterns."""
    late = df[df["dap"] >= 649]
    return late.groupby(["environment", "dap"], as_index=False)[["soil_k_mgkg", "tissue_k_pct"]].mean()
