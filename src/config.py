"""
Contains variables for everything
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = OUTPUT_DIR / "reports"

RAW_DATA_PATH = DATA_DIR / "AgriRiskFin_Dataset.csv"
CLEAN_DATA_PATH = DATA_DIR / "data_cleaned.csv"

# Scenario multipliers (your notebook values)
CARBON_PRICE_SCENARIOS_USD2010: Dict[str, float] = {
    "Delayed Transition": 10.0,
    "Net Zero(NZ) 2050": 110.0,
    "Divergent Net Zero": 300.0,
}

# Output generation defaults (your notebook choices)
DEFAULT_YEARS = 5
DEFAULT_GROWTH_RATE = 0.02
DEFAULT_DISCOUNT_RATE = 0.05

# Evaluation defaults
DEFAULT_RANDOM_STATE = 37
DEFAULT_CV_SPLITS = 5
TOP_RISK_Q = 0.90  # top 10%

@dataclass(frozen=True)
class PipelineConfig:
    random_state: int = DEFAULT_RANDOM_STATE
    cv_splits: int = DEFAULT_CV_SPLITS
    years: int = DEFAULT_YEARS
    growth_rate: float = DEFAULT_GROWTH_RATE
    discount_rate: float = DEFAULT_DISCOUNT_RATE
    top_risk_q: float = TOP_RISK_Q
