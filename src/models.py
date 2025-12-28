from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer

@dataclass
class ModelResult:
    model: Pipeline
    features: List[str]
    best_alpha: float
    test_rmse: float
    test_r2: float

def _rmse_scorer():
    return make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), greater_is_better=False)

def _to_int_bools(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    bool_cols = out.select_dtypes(include=["bool"]).columns
    out[bool_cols] = out[bool_cols].astype(int)
    return out

def tune_ridge_alpha(X: pd.DataFrame, y: pd.Series, random_state: int, cv_splits: int = 5) -> float:
    """
    Simple grid tuning like your notebook.
    """
    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    alphas = np.logspace(-3, 3, 20)
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    best_alpha, best_score = None, -np.inf
    for a in alphas:
        pipe.set_params(model__alpha=a)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=_rmse_scorer())
        mean_score = scores.mean()  # negative RMSE; higher is better
        if mean_score > best_score:
            best_score, best_alpha = mean_score, a
    return float(best_alpha)

def train_ridge_regression(df: pd.DataFrame, target: str, feature_exclude: List[str], random_state: int, cv_splits: int = 5) -> ModelResult:
    """
    Generic Ridge training with:
      - exclusion list (avoid leakage: Profit_Margin when predicting Net_Profit etc.)
      - alpha tuning via CV
      - test metrics
    """
    y = df[target]
    X = df.drop(columns=[c for c in feature_exclude if c in df.columns])

    X = _to_int_bools(X)

    best_alpha = tune_ridge_alpha(X, y, random_state=random_state, cv_splits=cv_splits)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=best_alpha))])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = float(r2_score(y_test, pred))

    return ModelResult(model=model, features=list(X.columns), best_alpha=best_alpha, test_rmse=rmse, test_r2=r2)

def predict_column(df: pd.DataFrame, model: Pipeline, features: List[str], out_col: str) -> pd.DataFrame:
    out = df.copy()
    X = out[features]
    X = _to_int_bools(X)
    out[out_col] = model.predict(X)
    return out
