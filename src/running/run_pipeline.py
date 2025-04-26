#!/usr/bin/env python3
"""
src/running/run_pipeline.py

Main entry point for the Navy Course Completion Prediction pipeline.
"""

import argparse
import logging
import numpy as np
import pandas as pd
import shap

# Logging config
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger()

from src.data.load_data import load_and_label
from src.features.engineer import engineer_features
from src.models.train import prepare_features, train_models, evaluate_model
from src.visualization.plots import (
    plot_confusion_matrices,
    plot_roc_curves,
    plot_shap_summary
)

def split_by_counter(data, col='assessment_counter', train=0.6, val=0.2):
    max_val = data[col].max()
    train_idx = int(max_val * train)
    val_idx   = int(max_val * (train + val))
    train     = data[data[col] <= train_idx]
    val       = data[(data[col] > train_idx) & (data[col] <= val_idx)]
    test      = data[data[col] > val_idx]
    return train, val, test

def run_stage(stage, data_fp, labels_fp):
    logger.info(f"### Stage {stage} pipeline start")

    # 1. Load & label
    df = load_and_label(data_fp, stage, labels_fp)
    logger.info(f"Loaded {len(df)} rows for Stage {stage}")

    # 2. Feature engineering
    df = engineer_features(df)
    logger.info("Feature engineering complete")

    # 3. Data split
    train_df, val_df, test_df = split_by_counter(df)
    logger.info(
        "Split rows → train=%d, val=%d, test=%d",
        len(train_df), len(val_df), len(test_df)
    )
    logger.info(
        "Unique assessment events → train=%d, val=%d, test=%d",
        train_df['assessment_counter'].nunique(),
        val_df['assessment_counter'].nunique(),
        test_df['assessment_counter'].nunique()
    )

    # 4. Prepare features & target
    X_train, y_train = train_df.drop(columns=['target']), train_df['target']
    X_test,  y_test  = test_df.drop(columns=['target']),  test_df['target']

    # 5. Feature selection
    X_train_sel = prepare_features(X_train, y_train)
    X_test_sel  = X_test[X_train_sel.columns]
    logger.info("Selected %d features: %s",
                X_train_sel.shape[1],
                list(X_train_sel.columns))

    # 6. Train & tune models
    models = train_models(X_train_sel, y_train)

    # 7. Evaluate all models
    for name, mdl in models.items():
        evaluate_model(name, mdl, X_test_sel, y_test)

    # 8. Baseline Hirustec heuristic
    if 'final_instructor_score' in test_df:
        preds = (test_df['final_instructor_score'] > 58).astype(int)
        logger.info("Evaluating Hirustec heuristic (score>58)")
        print("--- Hirustec baseline ---")
        print(classification_report(y_test, preds, zero_division=0))

    # 9. SHAP importances for RF
    if 'rf' in models:
        logger.info("Computing SHAP importances (Random Forest)")
        expl = shap.TreeExplainer(models['rf'])
        shap_vals = expl.shap_values(X_test_sel)
        vals = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        imp = pd.Series(np.abs(vals).mean(0), index=X_test_sel.columns)
        logger.info("Top 10 SHAP features → %s", imp.nlargest(10).to_dict())

    # 10. Plot results
    plot_confusion_matrices(models, X_test_sel, y_test)
    plot_roc_curves(models, X_test_sel, y_test)
    plot_shap_summary(models['rf'], X_test_sel)

def main():
    parser = argparse.ArgumentParser(
        description="Run Navy Course Completion Prediction"
    )
    parser.add_argument(
        "--data-file", "-d",
        default="navy_assessments_v1.xlsx",
        help="Path to the assessment data Excel file"
    )
    parser.add_argument(
        "--labels-file", "-l",
        default="navy_assessment.xlsx - labels.csv",
        help="Path to the labels CSV file"
    )
    parser.add_argument(
        "--stages", "-s",
        type=int, nargs="+", default=[1, 2],
        help="Stages to run, e.g. --stages 1 2"
    )
    args = parser.parse_args()

    for stg in args.stages:
        run_stage(stg, args.data_file, args.labels_file)

if __name__ == "__main__":
    main()
