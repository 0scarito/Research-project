from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd

def project_revenue_paths(df: pd.DataFrame, base_revenue_col: str, years: int, growth_rate: float) -> pd.DataFrame:
    out = df.copy()
    out["Revenue_Year_1"] = out[base_revenue_col]
    for year in range(2, years + 1):
        out[f"Revenue_Year_{year}"] = out[f"Revenue_Year_{year-1}"] * (1 + growth_rate)
    return out

def discounted_revenue_npv(df: pd.DataFrame, years: int, discount_rate: float) -> pd.Series:
    """
    NPV of projected revenue path. Assumes Revenue_Year_1..Revenue_Year_years exist.
    """
    npv = 0.0
    for year in range(1, years + 1):
        npv = npv + df[f"Revenue_Year_{year}"] / ((1 + discount_rate) ** year)
    return npv

def add_environmental_risk_index(df: pd.DataFrame, scenarios: Dict[str, float], emissions_proxy_col: str, out_prefix: str = "Environmental_Risk") -> pd.DataFrame:
    """
    Environmental risk index = emissions proxy * carbon price, shifted so min=0.
    """
    out = df.copy()
    for scenario, price in scenarios.items():
        col = f"{out_prefix}_{scenario}"
        out[col] = out[emissions_proxy_col] * price
        out[col] = out[col] - out[col].min()
    return out

def add_future_profit_index(df: pd.DataFrame, scenarios: Dict[str, float], future_revenue_col: str, emissions_proxy_col: str, out_prefix: str = "Future_Profit") -> pd.DataFrame:
    """
    Future profit index = predicted revenue (scaled) - carbon cost index (scaled).
    Note: This is an INDEX, not a unit-consistent $ profit, consistent with your notebook framing.
    """
    out = df.copy()

    # carbon cost based on future emissions proxy
    for scenario, price in scenarios.items():
        cost_col = f"Carbon_Cost_Future_{scenario}"
        out[cost_col] = out[emissions_proxy_col] * price
        out[cost_col] = out[cost_col] - out[cost_col].min()

        out[f"{out_prefix}_{scenario}"] = out[future_revenue_col] - out[cost_col]

    return out

def add_future_carbon_risk_index(df: pd.DataFrame, scenarios: Dict[str, float], future_revenue_col: str, out_prefix: str = "Carbon_Risk_Score_Future") -> pd.DataFrame:
    """
    Carbon risk index = carbon cost future - predicted revenue
    """
    out = df.copy()
    for scenario in scenarios:
        cost_col = f"Carbon_Cost_Future_{scenario}"
        out[f"{out_prefix}_{scenario}"] = out[cost_col] - out[future_revenue_col]
    return out

def add_stranded_flag(df: pd.DataFrame, scenario: str, risk_prefix: str = "Carbon_Risk_Score_Future") -> pd.DataFrame:
    out = df.copy()
    col = f"{risk_prefix}_{scenario}"
    out["Is_Stranded"] = out[col] > 0
    return out

def reconstruct_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct Region and Enterprise_Size from one-hot dummies (as in notebook).
    """
    out = df.copy()

    # Region
    out["Region"] = "East"
    if "Region_North" in out.columns: out.loc[out["Region_North"] == 1, "Region"] = "North"
    if "Region_South" in out.columns: out.loc[out["Region_South"] == 1, "Region"] = "South"
    if "Region_West"  in out.columns: out.loc[out["Region_West"]  == 1, "Region"] = "West"

    # Enterprise size
    out["Enterprise_Size"] = "Large"
    if "Enterprise_Size_Medium" in out.columns: out.loc[out["Enterprise_Size_Medium"] == 1, "Enterprise_Size"] = "Medium"
    if "Enterprise_Size_Small"  in out.columns: out.loc[out["Enterprise_Size_Small"]  == 1, "Enterprise_Size"] = "Small"

    return out

def add_climate_profile(df: pd.DataFrame, climate_stress_col: str = "Climate_Stress") -> pd.DataFrame:
    out = df.copy()
    q33 = out[climate_stress_col].quantile(0.33)
    q67 = out[climate_stress_col].quantile(0.67)
    out["Climate_Profile"] = pd.cut(
        out[climate_stress_col],
        bins=[-np.inf, q33, q67, np.inf],
        labels=["Low", "Medium", "High"]
    )
    return out
