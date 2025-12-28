from __future__ import annotations
from typing import List, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler

CATEGORICAL_COLS = ["Region", "Enterprise_Size", "Quarter"]

def drop_unneeded_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Enterprise_ID" in out.columns:
        out = out.drop(columns=["Enterprise_ID"])
    return out

def one_hot_encode(df: pd.DataFrame, categorical_cols: List[str] = None) -> pd.DataFrame:
    out = df.copy()
    categorical_cols = categorical_cols or CATEGORICAL_COLS
    out = pd.get_dummies(out, columns=categorical_cols, drop_first=True)
    return out

def scale_numeric(df: pd.DataFrame, exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Fits StandardScaler on numeric columns and returns scaled dataframe + scaler.
    """
    out = df.copy()
    exclude_cols = set(exclude_cols or [])
    num_cols = [c for c in out.columns if out[c].dtype in ["float64", "int64"] and c not in exclude_cols]

    scaler = StandardScaler()
    out[num_cols] = scaler.fit_transform(out[num_cols])
    return out, scaler

def build_clean_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Complete preprocessing: drop ID, add dummies, scale numerics.
    Keeps Financial_Risk_Level as a column for later analysis.
    """
    out = drop_unneeded_columns(df_raw)
    out = one_hot_encode(out)

    # Keep Financial_Risk_Level unscaled (it's categorical)
    exclude = ["Financial_Risk_Level"]
    out, _ = scale_numeric(out, exclude_cols=exclude)
    return out
