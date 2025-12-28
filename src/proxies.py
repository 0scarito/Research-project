from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd

def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

def build_emissions_proxy(
    df: pd.DataFrame,
    scale_col: str,
    input_cost_col: str = "Input_Cost_Index",
    climate_stress_col: str = "Climate_Stress",
    debt_col: str = "Debt_to_Equity",
    weights: Tuple[float, float, float, float] = (0.4, 0.2, 0.3, 0.1),
    use_log_scale: bool = True,
) -> pd.Series:
    w_scale, w_input, w_climate, w_debt = weights
    scale = df[scale_col].copy()
    if use_log_scale:
        scale = np.log1p(scale.clip(lower=0))

    return (
        w_scale * zscore(scale)
        + w_input * zscore(df[input_cost_col])
        + w_climate * zscore(df[climate_stress_col])
        + w_debt * zscore(df[debt_col])
    )

def add_proxy_variants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Emissions_Proxy_v1..v4 as in your notebook.
    """
    out = df.copy()
    out["Emissions_Proxy_v1"] = build_emissions_proxy(out, scale_col="Expenses", weights=(0.4, 0.2, 0.3, 0.1))
    out["Emissions_Proxy_v2"] = build_emissions_proxy(out, scale_col="Revenue",  weights=(0.4, 0.2, 0.3, 0.1))
    out["Emissions_Proxy_v3"] = build_emissions_proxy(out, scale_col="Expenses", weights=(0.25, 0.25, 0.25, 0.25))
    out["Emissions_Proxy_v4"] = build_emissions_proxy(out, scale_col="Revenue",  weights=(0.25, 0.25, 0.25, 0.25))
    return out

def mad(df: pd.DataFrame, a: str, b: str) -> float:
    return float(np.mean(np.abs(df[b] - df[a])))

def top_overlap(df: pd.DataFrame, a: str, b: str, q: float = 0.90) -> float:
    top_a = df[a] >= df[a].quantile(q)
    top_b = df[b] >= df[b].quantile(q)
    return float((top_a & top_b).sum() / max(1, top_a.sum()))

def compare_proxies(df: pd.DataFrame, q: float = 0.90) -> pd.DataFrame:
    proxies = ["Emissions_Proxy_v1", "Emissions_Proxy_v2", "Emissions_Proxy_v3", "Emissions_Proxy_v4"]
    rows = []
    for i in range(len(proxies)):
        for j in range(i + 1, len(proxies)):
            a, b = proxies[i], proxies[j]
            rows.append({
                "proxy_a": a,
                "proxy_b": b,
                "mad": mad(df, a, b),
                "top10_overlap": top_overlap(df, a, b, q=q),
            })
    return pd.DataFrame(rows)

def choose_baseline_proxy(df: pd.DataFrame, preferred: str = "Emissions_Proxy_v1") -> str:
    """
    Simple policy: choose a preferred proxy unless missing.
    You can later extend to choose based on stability table.
    """
    return preferred if preferred in df.columns else "Emissions_Proxy_v1"
