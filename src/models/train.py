"""Train each model, evaluate on the held-out fold, return a leaderboard."""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def _metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_and_evaluate(models, preprocessor, X_train, X_test, y_train, y_test):
    """Fit every model; CatBoost gets raw strings, the rest get encoded ints."""
    X_train_enc = preprocessor.transform(X_train, encode=True)
    X_test_enc = preprocessor.transform(X_test, encode=True)
    X_train_raw = preprocessor.transform(X_train, encode=False)
    X_test_raw = preprocessor.transform(X_test, encode=False)
    cat_cols = preprocessor.cat_cols_

    rows, fitted = [], {}
    for name, spec in models.items():
        est = spec["estimator"]
        logger.info("Training %s", name)
        t0 = time.time()
        if spec["native_categoricals"]:
            fit_kwargs = {"cat_features": cat_cols} if cat_cols else {}
            est.fit(X_train_raw, y_train, **fit_kwargs)
            pred = est.predict(X_test_raw)
        else:
            est.fit(X_train_enc, y_train)
            pred = est.predict(X_test_enc)
        m = _metrics(y_test, pred)
        m.update({"Model": name, "Train_Time_Seconds": round(time.time() - t0, 2)})
        rows.append(m)
        fitted[name] = est
        logger.info("%s: R2=%.4f RMSE=%.2f MAE=%.2f", name, m["R2"], m["RMSE"], m["MAE"])

    leaderboard = (
        pd.DataFrame(rows)
        .sort_values("R2", ascending=False)
        .reset_index(drop=True)
    )
    return leaderboard, fitted
