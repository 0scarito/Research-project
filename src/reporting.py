from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

def save_table_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def save_table_md(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    md = df.to_markdown(index=index)
    path.write_text(md, encoding="utf-8")

def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def format_eda_summary(summary: Any) -> str:
    return (
        f"# EDA Summary\n\n"
        f"- Rows: {summary.n_rows}\n"
        f"- Columns: {summary.n_cols}\n"
        f"- Duplicates: {summary.n_duplicates}\n"
        f"- Missing values (total): {sum(summary.missing_by_col.values())}\n"
    )
