"""Machine-learning models with grouped validation to avoid plant-level leakage."""

from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["treatment_t_acre", "dap", "temperature_c", "rainfall_mm", "soil_n_mgkg", "soil_p_mgkg", "soil_k_mgkg", "tissue_n_pct", "tissue_p_pct", "tissue_k_pct", "ndvi", "ndre", "gndvi", "red_edge_slope", "nir_mean", "swir_mean"]
CATEGORICAL_FEATURES = ["environment", "species"]


def build_preprocessor():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)])


def evaluate_stress_classifier(df: pd.DataFrame, n_splits: int = 5):
    x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["stress_flag"].astype(int)
    groups = df["plant_id"]
    cv = GroupKFold(n_splits=n_splits)
    model = Pipeline([("prep", build_preprocessor()), ("model", RandomForestClassifier(n_estimators=350, min_samples_leaf=3, class_weight="balanced", random_state=42))])
    pred = cross_val_predict(model, x, y, groups=groups, cv=cv, method="predict")
    metrics = {"model": "RandomForestClassifier", "target": "stress_flag", "validation": "GroupKFold by plant_id", "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0)}
    cm = confusion_matrix(y, pred)
    model.fit(x, y)
    prep = model.named_steps["prep"]
    importance = pd.DataFrame({"feature": list(prep.get_feature_names_out()), "importance": model.named_steps["model"].feature_importances_}).sort_values("importance", ascending=False).reset_index(drop=True)
    return model, metrics, cm, importance


def evaluate_flower_regressor(df: pd.DataFrame, n_splits: int = 5):
    x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["flower_count"].astype(float)
    groups = df["plant_id"]
    cv = GroupKFold(n_splits=n_splits)
    model = Pipeline([("prep", build_preprocessor()), ("model", RandomForestRegressor(n_estimators=350, min_samples_leaf=2, random_state=42))])
    pred = cross_val_predict(model, x, y, groups=groups, cv=cv, method="predict")
    metrics = {"model": "RandomForestRegressor", "target": "flower_count", "validation": "GroupKFold by plant_id", "mae": mean_absolute_error(y, pred), "r2": r2_score(y, pred)}
    model.fit(x, y)
    return model, metrics, pred
