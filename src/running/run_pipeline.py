import argparse
import logging
from sklearn.metrics import classification_report, f1_score, accuracy_score
import mlflow.sklearn

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger()

from src.data.load_data import load_and_label
from src.features.engineer import engineer_features
from src.models.train import prepare_features, train_models, evaluate_model
from src.visualization.plots import (
    plot_confusion_matrices, plot_roc_curves, plot_shap_summary
)


def split_data(df, train_frac=0.8):
    df_shuffled = df.sample(frac=1, random_state=42)
    cut = int(len(df_shuffled) * train_frac)
    return df_shuffled[:cut], df_shuffled[cut:]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', '-d', default='navy_assessments.csv')
    parser.add_argument('--labels-file', '-l', default='labels.csv')
    parser.add_argument('--stage', '-s', type=int, choices=[1,2], required=True)
    args = parser.parse_args()

    mlflow.set_experiment('navy_completion_prediction')
    with mlflow.start_run(run_name=f'stage_{args.stage}'):
        mlflow.log_param('stage', args.stage)

        # Load & label
        df = load_and_label(args.data_file, args.stage, args.labels_file)
        logger.info(f"Stage {args.stage}: {len(df)} records")
        mlflow.log_param('num_records', len(df))

        # Feature engineering
        df_feat = engineer_features(df)
        mlflow.log_param('num_features_before_selection', df_feat.shape[1] - 2)

        # Split
    train_df, test_df = split_data(df_feat)
    X_train = train_df.drop(columns=['target', 'final_instructor_score'])
    y_train = train_df['target']
    X_test  = test_df.drop(columns=['target', 'final_instructor_score'])
    y_test  = test_df['target']
    mlflow.log_param('train_size', len(train_df))
    mlflow.log_param('test_size', len(test_df))

    # Feature selection
    X_train_sel = prepare_features(X_train, y_train)
    X_test_sel  = X_test[X_train_sel.columns]
    mlflow.log_param('num_features_selected', X_train_sel.shape[1])

    # Print selected features
    logger.info("Selected features for Stage %d: %s", args.stage, list(X_train_sel.columns))

    # Train models
    models = train_models(X_train_sel, y_train)

    # Feature importance / coefficients
    for name, mdl in models.items():
        if hasattr(mdl, 'feature_importances_'):
            importances = mdl.feature_importances_
            feat_imp = sorted(zip(X_train_sel.columns, importances), key=lambda x: x[1], reverse=True)
            logger.info(f"{name} feature importances: %s", feat_imp)
            mlflow.log_text(str(feat_imp), f"feature_importances_{name}.txt")
        elif hasattr(mdl, 'coef_'):
            coefs = mdl.coef_[0]
            feat_imp = sorted(zip(X_train_sel.columns, coefs), key=lambda x: abs(x[1]), reverse=True)
            logger.info(f"{name} coefficients: %s", feat_imp)
            mlflow.log_text(str(feat_imp), f"coefficients_{name}.txt")

            # Evaluate
            for name, mdl in models.items():
                evaluate_model(name, mdl, X_test_sel, y_test)

            # Baseline heuristic (comparison only, not fed to model)
            baseline = (test_df['final_instructor_score'] > 58).astype(int)
            print("--- Baseline Heuristic ---")
            print(classification_report(y_test, baseline, zero_division=0))

            # Visualize results
            plot_confusion_matrices(models, X_test_sel, y_test)
            plot_roc_curves(models, X_test_sel, y_test)
            plot_shap_summary(models['ensemble'], X_test_sel)
            var = (models['ensemble'], X_test_sel)