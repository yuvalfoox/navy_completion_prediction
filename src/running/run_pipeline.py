#!/usr/bin/env python3

import argparse
import logging
import warnings
import pandas as pd
import shap
import numpy as np

from src.data.load_data import load_data_and_label
from src.features.engineer import engineer_features
from src.models.train import prepare_features, train_models, evaluate_model
from src.visualization.plots import plot_confusion_matrices, plot_roc_curves, plot_shap_summary

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def split_by_date(data: pd.DataFrame, train_frac=0.6, val_frac=0.2):
    """Split chronologically based on main_assessment_date."""
    data = data.sort_values('main_assessment_date')
    n = len(data)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return data.iloc[:train_end], data.iloc[train_end:val_end], data.iloc[val_end:]

def run_stage(stage: int, data_fp: str, labels_fp: str):
    logger.info(f"### Stage {stage} pipeline start")

    # 1. Load & label
    df = load_data_and_label(data_fp, labels_fp, stage)
    logger.info(f"Loaded {len(df)} rows for Stage {stage}")

    # 2. Feature engineering
    df = engineer_features(df)
    logger.info("Feature engineering complete")

    # 3. Split chronologically
    train_df, val_df, test_df = split_by_date(df)

    # Drop only main_assessment_date
    drop_cols = ['main_assessment_date']
    train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    val_df   = val_df.drop(columns=[c for c in drop_cols if c in val_df.columns])
    test_df  = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    logger.info("Train/Val/Test split: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))

    # Prepare matrices
    X_train = train_df.drop(columns=['target', 'label_score'])
    y_train = train_df['target']
    X_val   = val_df.drop(columns=['target', 'label_score'])
    y_val   = val_df['target']
    X_test  = test_df.drop(columns=['target', 'label_score'])
    y_test  = test_df['target']

    # 🚨 Baseline evaluation before modeling
    if 'naval_officers_formula_score' in test_df.columns:
        from sklearn.metrics import classification_report
        preds = (test_df['naval_officers_formula_score'] >= 58).astype(int)
        print("\n--- Baseline: naval_officers_formula_score >= 58 ---")
        print(classification_report(y_test, preds, zero_division=0))

    # Feature selection
    X_train_sel = prepare_features(X_train, y_train)
    X_val_sel   = X_val[X_train_sel.columns]
    X_test_sel  = X_test[X_train_sel.columns]

    logger.info("Selected features: %s", list(X_train_sel.columns))
    print("Selected features:", list(X_train_sel.columns))

    # Train models using train/val sets
    models = train_models(X_train_sel, y_train, X_val_sel, y_val)

    # Evaluate models
    for name, model in models.items():
        evaluate_model(name, model, X_test_sel, y_test)

    logger.info("Computing SHAP importances for Random Forest")
    explainer = shap.TreeExplainer(models['rf'])
    shap_values = explainer.shap_values(X_test_sel)

    # --- Correct handling ---
    if isinstance(shap_values, list):
        vals = shap_values[1]  # List → pick class 1
    elif shap_values.ndim == 3:
        vals = shap_values[:, :, 1]  # 3D array → pick class 1
    else:
        vals = shap_values  # Already 2D

    # Aggregate SHAP values per feature
    mean_importances = np.abs(vals).mean(axis=0)

    # Now it's safe to create a Series
    imp = pd.Series(mean_importances, index=X_test_sel.columns)
    logger.info("Top 10 SHAP features: %s", imp.nlargest(10).to_dict())

    plot_confusion_matrices(models, X_test_sel, y_test)
    plot_roc_curves(models, X_test_sel, y_test)
    if 'rf' in models:
        plot_shap_summary(models['rf'], X_test_sel)


def main():
    parser = argparse.ArgumentParser(description="Run Navy Completion Prediction pipeline")
    parser.add_argument('--data-file', required=True)
    parser.add_argument('--labels-file', required=True)
    parser.add_argument('--stage', type=int, choices=[1, 2], help="Stage 1 or Stage 2")
    args = parser.parse_args()

    stages = [args.stage] if args.stage else [1, 2]
    for stg in stages:
        run_stage(stg, args.data_file, args.labels_file)


if __name__ == "__main__":
    main()
