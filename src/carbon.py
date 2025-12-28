from __future__ import annotations
from typing import Dict
import pandas as pd

def add_carbon_cost_index(df: pd.DataFrame, scenarios: Dict[str, float], proxy_col: str, prefix: str = "Carbon_Cost") -> pd.DataFrame:
    """
    Carbon cost index = proxy * price, then shifted so min=0 (proxy is z-scored).
    """
    out = df.copy()
    for scenario, price in scenarios.items():
        col = f"{prefix}_{scenario}"
        out[col] = out[proxy_col] * price

    for scenario in scenarios:
        col = f"{prefix}_{scenario}"
        out[col] = out[col] - out[col].min()
    return out

def add_adjusted_profit(df: pd.DataFrame, scenarios: Dict[str, float], profit_col: str, cost_prefix: str = "Carbon_Cost", out_prefix: str = "Adj_Profit") -> pd.DataFrame:
    out = df.copy()
    for scenario in scenarios:
        out[f"{out_prefix}_{scenario}"] = out[profit_col] - out[f"{cost_prefix}_{scenario}"]
    return out

def add_carbon_risk_score(df: pd.DataFrame, scenarios: Dict[str, float], profit_col: str, cost_prefix: str = "Carbon_Cost", out_prefix: str = "Carbon_Risk_Score") -> pd.DataFrame:
    out = df.copy()
    for scenario in scenarios:
        out[f"{out_prefix}_{scenario}"] = out[f"{cost_prefix}_{scenario}"] - out[profit_col]
    return out
