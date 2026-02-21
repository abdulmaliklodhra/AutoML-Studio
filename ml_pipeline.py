"""
ml_pipeline.py
--------------
Backend ML logic for the AutoML PyCaret application.
Handles task-type detection and runs PyCaret pipelines.
"""

import pandas as pd
import numpy as np
import os
import sys


def detect_task_type(df: pd.DataFrame, target: str) -> str:
    """
    Auto-detect whether the problem is classification or regression.
    
    Rules:
    - If target dtype is object/bool/category → classification
    - If unique values <= 20 (and integer) → likely classification
    - Otherwise → regression
    """
    col = df[target].dropna()
    n_unique = col.nunique()
    dtype = col.dtype

    if dtype == "object" or dtype.name == "category" or dtype == bool:
        return "classification"
    if pd.api.types.is_integer_dtype(dtype) and n_unique <= 20:
        return "classification"
    if pd.api.types.is_float_dtype(dtype) and n_unique <= 15:
        return "classification"
    return "regression"


def run_classification_pipeline(df: pd.DataFrame, target: str):
    """
    Run full PyCaret classification pipeline.
    Returns: (best_model, final_model, compare_df, setup_df)
    """
    # Import inside function to avoid namespace conflicts
    from pycaret.classification import (
        setup, compare_models, finalize_model,
        pull, save_model, plot_model
    )

    # Step 1: Setup
    setup(
        data=df,
        target=target,
        session_id=42,
        verbose=False,
        html=False,
    )
    setup_df = pull()

    # Step 2: Compare models
    best_model = compare_models(verbose=False)
    compare_df = pull()

    # Step 3: Finalize model
    final_model = finalize_model(best_model)

    return best_model, final_model, compare_df, setup_df


def run_regression_pipeline(df: pd.DataFrame, target: str):
    """
    Run full PyCaret regression pipeline.
    Returns: (best_model, final_model, compare_df, setup_df)
    """
    from pycaret.regression import (
        setup, compare_models, finalize_model,
        pull, save_model, plot_model
    )

    # Step 1: Setup
    setup(
        data=df,
        target=target,
        session_id=42,
        verbose=False,
        html=False,
    )
    setup_df = pull()

    # Step 2: Compare models
    best_model = compare_models(verbose=False)
    compare_df = pull()

    # Step 3: Finalize model
    final_model = finalize_model(best_model)

    return best_model, final_model, compare_df, setup_df


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importance if the model supports it.
    Returns DataFrame with Feature and Importance columns.
    """
    try:
        # Try to get feature importances from the final estimator
        estimator = model
        if hasattr(model, 'steps'):
            # It's a pipeline
            estimator = model.steps[-1][1]
        
        if hasattr(estimator, 'feature_importances_'):
            importances = estimator.feature_importances_
        elif hasattr(estimator, 'coef_'):
            coef = estimator.coef_
            if coef.ndim > 1:
                importances = np.abs(coef).mean(axis=0)
            else:
                importances = np.abs(coef)
        else:
            return None

        # Match lengths — PyCaret transforms features, so use minimum
        min_len = min(len(importances), len(feature_names))
        fi_df = pd.DataFrame({
            'Feature': feature_names[:min_len],
            'Importance': importances[:min_len]
        }).sort_values('Importance', ascending=False)
        return fi_df
    except Exception:
        return None


def get_classification_metric_cols() -> list:
    """Return the primary metric columns for classification."""
    return ['Model', 'Accuracy', 'AUC', 'Recall', 'Prec.', 'F1', 'Kappa', 'MCC']


def get_regression_metric_cols() -> list:
    """Return the primary metric columns for regression."""
    return ['Model', 'MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE']


def save_final_model(model, task_type: str, output_dir: str = ".") -> str:
    """Save the finalized model to disk."""
    path = os.path.join(output_dir, f"best_{task_type}_model")
    try:
        if task_type == "classification":
            from pycaret.classification import save_model
        else:
            from pycaret.regression import save_model
        save_model(model, path)
        return path + ".pkl"
    except Exception as e:
        return None
