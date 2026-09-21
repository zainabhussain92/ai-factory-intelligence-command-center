"""Stage V — Predictive Maintenance Agent.

Uses Stage II's already-trained model to score a machine's most recent
sensor window. This is inference only - the model is loaded from
`models/gru_model.npz` (the model Stage II itself selected as best by
validation F1, per `reports/stage2_model_comparison.json`) and never
retrained or modified here.

No per-machine "live" prediction artifact existed in Stage II's output
(only aggregate validation/test metrics), so scoring a specific machine's
current window is new Stage V integration work - not a rebuild of Stage II,
since it reuses the exact trained weights, feature contract, and scaling
statistics Stage II produced.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.config import PROCESSED_DIR, REPORTS_DIR, PROJECT_ROOT
from src.deep_learning.gru_numpy import GRUBinaryClassifier
from src.agents.schemas import unavailable_result, error_result

MODELS_DIR = PROJECT_ROOT / "models"
SHIFT_ORDER = {"Morning": 0, "Afternoon": 1, "Night": 2}


@lru_cache(maxsize=1)
def _load_gru_threshold() -> float:
    metrics_path = REPORTS_DIR / "stage2_gru_metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())["selected_threshold"]
    return 0.5  # documented fallback if the metrics report is missing


def _probability_to_risk_level(prob: float, threshold: float) -> str:
    """Risk banding around the model's own validation-tuned threshold
    (never refit here) - HIGH at/above threshold, LOW below half the
    threshold, MEDIUM between. A documented domain rule, not a new fit on
    validation/test data.
    """
    if prob >= threshold:
        return "HIGH"
    if prob < threshold / 2:
        return "LOW"
    return "MEDIUM"


def _latest_window(machine_id: str, window: int) -> pd.DataFrame | None:
    """Most recent `window` consecutive shifts for a machine, preferring
    the test split (most recent chronological data); never spans machines.
    """
    for split_name in ["test.csv", "validation.csv", "train.csv"]:
        path = PROCESSED_DIR / split_name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        g = df[df["machine_id"] == machine_id].copy()
        if len(g) < window:
            continue
        g["_shift_order"] = g["shift"].map(SHIFT_ORDER)
        g = g.sort_values(["sim_day", "_shift_order"])
        return g.tail(window).reset_index(drop=True)
    return None


@lru_cache(maxsize=1)
def _load_gru_artifacts():
    """Load the trained GRU model + its feature contract (read-only, no
    retraining). Returns (model, contract) or (None, None) if missing.

    Cached with lru_cache: this is called from three places per analysis
    run (Predictive Maintenance Agent, Digital Twin simulator, XAI GRU
    explainer). Without caching, the same .npz weights + JSON contract are
    re-read from disk on every single call, which is the main cause of a
    slow "Run Full Analysis" click. The underlying files never change while
    the app is running, so caching them for the lifetime of the process is
    safe - it does not change any prediction, only how often the same,
    already-trained artifact is re-loaded from disk.
    """
    contract_path = MODELS_DIR / "gru_feature_contract.json"
    model_path = MODELS_DIR / "gru_model.npz"
    if not contract_path.exists() or not model_path.exists():
        return None, None
    contract = json.loads(contract_path.read_text())
    model = GRUBinaryClassifier.load(model_path)
    return model, contract


def build_gru_input(machine_id: str, contract: dict):
    """Build the exact (1, window, n_features) standardized sequence input
    the GRU expects for a machine's most recent window, plus the raw
    (unstandardized) window rows for reference. Shared by the Predictive
    Maintenance Agent and Stage VI's XAI layer so both explain the same
    input construction - not two parallel implementations.

    Returns (X_seq, recent_df) or (None, None) if too little history exists.
    """
    window = contract["window"]
    status_cols = contract["status_cols"]

    recent = _latest_window(machine_id, window)
    if recent is None:
        return None, None

    recent = recent.copy()
    for c, med in contract["impute_medians"].items():
        if c in recent.columns:
            recent[c] = recent[c].fillna(med)

    status_dummies = pd.get_dummies(recent["operating_status"], prefix="operating_status")
    status_dummies = status_dummies.reindex(columns=status_cols, fill_value=0)
    for c in status_cols:
        recent[c] = status_dummies[c].values

    feature_cols = contract["feature_cols"]
    means = pd.Series(contract["means"])
    stds = pd.Series(contract["stds"])
    X = recent[feature_cols].astype(float)
    X = (X - means[feature_cols]) / stds[feature_cols]
    X_seq = X.to_numpy().reshape(1, window, len(feature_cols))
    return X_seq, recent


def get_prediction(machine_id: str) -> dict:
    model, contract = _load_gru_artifacts()
    if model is None:
        return unavailable_result("Stage II GRU model/feature contract not found")

    numeric_cols = [c for c in contract["feature_cols"] if c not in contract["status_cols"]]

    try:
        X_seq, recent = build_gru_input(machine_id, contract)
        if X_seq is None:
            return unavailable_result(
                f"Fewer than {contract['window']} recorded shifts available for machine {machine_id}")
        prob = float(model.predict_proba(X_seq)[0])
    except Exception as exc:  # never silently substitute a fake result
        return error_result(f"GRU inference failed for {machine_id}: {exc}")

    threshold = _load_gru_threshold()
    risk_level = _probability_to_risk_level(prob, threshold)

    # Which raw sensor channels are most deviated (|z-score|) in the most
    # recent reading - a simple, transparent signal-ranking derived here at
    # inference time, not a claim of a Stage II feature-importance model.
    # Stage VI (src/xai/gru_explainer.py) provides a rigorous permutation-
    # importance version of this same question for the full window.
    means = pd.Series(contract["means"])
    stds = pd.Series(contract["stds"])
    last_row = recent[numeric_cols].iloc[-1]
    z_scores = ((last_row - means[numeric_cols]) / stds[numeric_cols]).abs()
    important_signals = z_scores.sort_values(ascending=False).head(3).index.tolist()

    return {
        "status": "ok",
        "machine_id": machine_id,
        "failure_probability": round(prob, 4),
        "predicted_failure": int(prob >= threshold),
        "risk_level": risk_level,
        "model_name": "gru",
        "decision_threshold": threshold,
        "important_signals": important_signals,
        "window_shifts_used": contract["window"],
        "source": "models/gru_model.npz (Stage II's selected-best model, inference only)",
    }
