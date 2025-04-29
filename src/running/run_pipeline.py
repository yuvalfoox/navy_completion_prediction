#!/usr/bin/env python3

import argparse
import logging
import warnings
import os
import json
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, f1_score

from src.data.load_data import load_data_and_label
from src.features.engineer import engineer_features
from src.models.train import prepare_features, train_models, evaluate_model
from src.config import train_frac, val_frac

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def prepare_X_y(df):
    drop_cols = ['target']
    if 'label_score' in df.columns:
        drop_cols.append('label_score')
    X = df.drop(columns=drop_cols)
    y = df['target']
    return X, y


def split_by_date(data: pd.DataFrame):
    """Chronological split by main_assessment_date into train/val/test."""
    data = data.sort_values('main_assessment_date').reset_index(drop=True)
    data['row_index'] = np.arange(1, len(data) + 1)

    n = len(data)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = data.iloc[:train_end]
    val_df = data.iloc[train_end:val_end]
    test_df = data.iloc[val_end:]

    print(f"Train rows: {train_df['row_index'].min()} - {train_df['row_index'].max()} ({len(train_df)})")
    print(f"Val rows: {val_df['row_index'].min()} - {val_df['row_index'].max()} ({len(val_df)})")
    print(f"Test rows: {test_df['row_index'].min()} - {test_df['row_index'].max()} ({len(test_df)})")

    return train_df.drop(columns=['row_index']), val_df.drop(columns=['row_index']), test_df.drop(columns=['row_index'])


def run_stage(stage: int, data_fp: str, labels_fp: str):
    logger.info(f"### Stage {stage} pipeline start")

    df = load_data_and_label(data_fp, labels_fp, stage)
    logger.info(f"Loaded {len(df)} rows for Stage {stage}")

    df = engineer_features(df)
    logger.info("Feature engineering complete")

    train_df, val_df, test_df = split_by_date(df)

    drop_cols = ['main_assessment_date']
    train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    val_df = val_df.drop(columns=[c for c in drop_cols if c in val_df.columns])
    test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    logger.info("Train/Val/Test split: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))

    X_train, y_train = prepare_X_y(train_df)
    X_val, y_val = prepare_X_y(val_df)
    X_test, y_test = prepare_X_y(test_df)

    output_dir = f"output/stage{stage}"
    os.makedirs(output_dir, exist_ok=True)

    # --- Heuristic Baseline ---
    if 'naval_officers_formula_score' in test_df.columns:
        preds = (test_df['naval_officers_formula_score'] >= 58).astype(int)

        f1 = f1_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, preds).tolist()
        fpr, tpr, _ = roc_curve(y_test, preds)
        prec, rec, _ = precision_recall_curve(y_test, preds)

        heuristic_metrics = {
            "f1": f1,
            "precision": report['1']['precision'],
            "recall": report['1']['recall'],
            "best_threshold": 0.5
        }

        heuristic_file = os.path.join(output_dir, "heuristic_metrics.json")
        with open(heuristic_file, "w") as f:
            json.dump({"heuristic": heuristic_metrics}, f, indent=4)

        with open(os.path.join(output_dir, "confusion_matrix_heuristic.json"), "w") as f:
            json.dump(cm, f, indent=4)

        with open(os.path.join(output_dir, "roc_curve_heuristic.json"), "w") as f:
            json.dump({"fpr": fpr.tolist(), "tpr": tpr.tolist()}, f, indent=4)

        with open(os.path.join(output_dir, "prc_curve_heuristic.json"), "w") as f:
            json.dump({"precision": prec.tolist(), "recall": rec.tolist()}, f, indent=4)

        with open(os.path.join(output_dir, "threshold_scores_heuristic.json"), "w") as f:
            json.dump({"0.5": f1}, f, indent=4)

        print(f"\n--- Heuristic (naval_officers_formula_score >= 58) ---")
        print(classification_report(y_test, preds, zero_division=0))

    # --- Train ML Models ---
    X_train_sel, X_val_sel, X_test_sel = prepare_features(X_train, y_train, X_val, X_test)
    logger.info("Selected features: %s", list(X_train_sel.columns))

    models = train_models(X_train_sel, y_train, X_val_sel, y_val)

    for name, model in models.items():
        evaluate_model(name, model, X_test_sel, y_test, output_dir)
        logger.info(f"Model {name} evaluation complete")

    # --- Merge Heuristic + Model Metrics ---
    metrics_file = os.path.join(output_dir, "models_metrics.json")
    heuristic_file = os.path.join(output_dir, "heuristic_metrics.json")

    if os.path.exists(metrics_file) and os.path.exists(heuristic_file):
        with open(metrics_file, "r") as f:
            metrics_models = json.load(f)

        with open(heuristic_file, "r") as f:
            metrics_heuristic = json.load(f)

        combined = {**metrics_models, **metrics_heuristic}
        with open(metrics_file, "w") as f:
            json.dump(combined, f, indent=4)

        os.remove(heuristic_file)


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
