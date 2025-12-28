"""
Inputs raw data from a path, outputs clean data to a path. Also ensures a directory exists before saving there
"""
from pathlib import Path
import pandas as pd

# this file contains the functions related to input/output. Even if those functions are only one line in length, they should be there for convenience

# loads original raw data into a Pandas dataframe, so that we can then use the ETL pipeline on it
def load_raw_data(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

# saves the cleaned-up Pandas dataframe to a csv file in a chosen filepath
def save_clean_data(df, path, index=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)

def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
