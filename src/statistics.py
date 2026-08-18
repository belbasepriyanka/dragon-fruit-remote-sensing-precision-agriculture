"""Statistical summaries for field-experiment style data."""

from __future__ import annotations
import pandas as pd


def grouped_descriptives(df: pd.DataFrame, response: str) -> pd.DataFrame:
    return df.groupby(["environment", "species", "treatment_t_acre"])[response].agg(["mean", "std", "count"]).reset_index()


def correlation_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[columns].corr(numeric_only=True)
