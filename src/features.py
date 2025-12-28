from __future__ import annotations
import pandas as pd
import numpy as np

def add_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Profit_Margin, Cost_Ratio, Debt_Ratio based on notebook definitions.
    """
    out = df.copy()
    out["Profit_Margin"] = out["Net_Profit"] / out["Revenue"]
    out["Cost_Ratio"] = out["Expenses"] / out["Revenue"]
    out["Debt_Ratio"] = out["Loan_Amount"] / out["Revenue"]
    return out

def add_climate_stress(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Climate_Stress (your notebook version):
      0.4 * Drought + 0.4 * Flood + 0.2 * z(Temperature deviation)
    """
    out = df.copy()
    temp_z = (out["Avg_Temperature"] - out["Avg_Temperature"].mean()) / (out["Avg_Temperature"].std() + 1e-9)
    out["Climate_Stress"] = (
        0.4 * out["Drought_Index"]
        + 0.4 * out["Flood_Risk_Score"]
        + 0.2 * temp_z
    )
    return out
