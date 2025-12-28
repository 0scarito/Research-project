from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@dataclass
class EDASummary:
    n_rows: int
    n_cols: int
    dtypes: Dict[str, str]
    missing_by_col: Dict[str, int]
    n_duplicates: int

def compute_eda_summary(df: pd.DataFrame) -> EDASummary:
    return EDASummary(
        n_rows=len(df),
        n_cols=df.shape[1],
        dtypes={c: str(t) for c, t in df.dtypes.items()},
        missing_by_col=df.isna().sum().to_dict(),
        n_duplicates=int(df.duplicated().sum()),
    )

def save_describe_table(df: pd.DataFrame, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.describe(include="all").to_csv(out_csv)

def plot_histograms(df: pd.DataFrame, cols: List[str], out_png: Path, bins: int = 30) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    df[cols].hist(bins=bins, figsize=(14, 10))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_corr_heatmap(df: pd.DataFrame, out_png: Path) -> None:
    """
    Simple matplotlib correlation heatmap (no seaborn needed).
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    num_df = df.select_dtypes(include=["float64", "int64"])
    corr = num_df.corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(corr.values)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
    plt.colorbar()
    plt.title("Correlation matrix (numeric features)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_scatter(df: pd.DataFrame, x: str, y: str, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x], df[y], s=10, alpha=0.5)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{y} vs {x}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
