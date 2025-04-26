#!/usr/bin/env python3
"""
src/running/run_pipeline.py

Main entry point for the Navy Course Completion Prediction pipeline.
"""

import argparse
import logging

# Set up root logger
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
    return (
        data[data[col] <= train_idx],
        data[(data[col] > train_idx) & (data[col] <= val_idx)],
        data[data[col] > val_idx]
    )

def run_stage(stage, filepath):
    logger.info(f"### Stage {stage} pipeline start")
    df = load_and_label(filepath, stage=stage)
    logger.info(f"Loaded {len(df)} records for Stage {stage}")

    df = engineer_features(df)
    logger.info("Feature engineering complete")

    train_df, val_df, test_df = split_by_counter(df)
    logger.info(f"Data split → train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    X_train, y_train = train_df.drop(columns=['target']), train_df['target']
    X_test,  y_test  = test_df.drop(columns=['target']), test_df['target']

    X_train_sel = prepare_features(X_train, y_train)
    X_test_sel  = X_test[X_train_sel.columns]
    logger.info(f"Feature selection complete: {X_train_sel.shape[1]} features")

    models = train_models(X_train_sel, y_train)

    # Evaluate each
    for name, model in models.items():
        evaluate_model(name, model, X_test_sel, y_test)

    # Visualize
    plot_confusion_matrices(models, X_test_sel, y_test)
    plot_roc_curves(models, X_test_sel, y_test)
    plot_shap_summary(models['rf'], X_test_sel)

def main():
    parser = argparse.ArgumentParser(
        description="Run Navy Course Completion Prediction pipeline"
    )
    parser.add_argument(
        "--filepath", "-f",
        type=str, default="navy_assessments_v1.xlsx",
        help="Path to the Excel file"
    )
    parser.add_argument(
        "--stages", "-s",
        type=int, nargs="+", default=[1, 2],
        help="Stages to run (e.g. --stages 1 2)"
    )
    args = parser.parse_args()

    for stage in args.stages:
        run_stage(stage, args.filepath)

if __name__ == "__main__":
    main()
