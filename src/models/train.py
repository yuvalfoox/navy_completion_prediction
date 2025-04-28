import logging
import pandas as pd
import numpy as np
import os
import json

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.metrics import classification_report, f1_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------- Feature Preparation ---------------------- #

def prepare_features(X_train: pd.DataFrame, y_train: pd.Series, X_val=None, X_test=None):
    logger.info("Preparing features: imputing missing values")

    imp = SimpleImputer(strategy='median')
    X_train_imp = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns)

    if X_val is not None:
        X_val_imp = pd.DataFrame(imp.transform(X_val), columns=X_val.columns)
    else:
        X_val_imp = None

    if X_test is not None:
        X_test_imp = pd.DataFrame(imp.transform(X_test), columns=X_test.columns)
    else:
        X_test_imp = None

    logger.info("Filtering features with >30% zeros")
    zero_frac = (X_train_imp == 0).mean()
    keep_mask = zero_frac < 0.30
    X_filt = X_train_imp.loc[:, keep_mask]

    logger.info("Applying low-variance filter")
    vt = VarianceThreshold(threshold=0.01)
    X_vt = pd.DataFrame(vt.fit_transform(X_filt), columns=X_filt.columns[vt.get_support()])

    logger.info("Computing mutual information for feature selection")
    mi = mutual_info_classif(X_vt, y_train, random_state=42)
    keep_mi = mi > 0.001
    X_sel = X_vt.loc[:, keep_mi]

    selected_features = X_sel.columns.tolist()

    if X_val_imp is not None:
        X_val_sel = X_val_imp[selected_features]
    else:
        X_val_sel = None

    if X_test_imp is not None:
        X_test_sel = X_test_imp[selected_features]
    else:
        X_test_sel = None

    logger.info(f"Selected {X_sel.shape[1]} features after MI filtering")
    return X_sel, X_val_sel, X_test_sel

# ---------------------- Model Trainers ---------------------- #

def tune_logistic_regression(X_train, y_train):
    logger.info("Tuning Logistic Regression")
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, solver='saga', random_state=42, class_weight='balanced'))
    ])
    param_grid = {
        'clf__C': np.logspace(-3, 2, 10),
        'clf__penalty': ['l1', 'l2'],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        lr_pipe, param_grid, n_iter=30, scoring='f1', cv=cv, n_jobs=-1, random_state=42
    )
    search.fit(X_train, y_train)
    logger.info(f"Best LR params: {search.best_params_}")
    return search.best_estimator_

def tune_random_forest(X_train, y_train):
    logger.info("Tuning Random Forest")
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [5, 10, 20, None],
        'min_samples_leaf': [1, 2, 5]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        rf, param_grid, n_iter=30, scoring='f1', cv=cv, n_jobs=-1, random_state=42
    )
    search.fit(X_train, y_train)
    logger.info(f"Best RF params: {search.best_params_}")
    return search.best_estimator_

def train_gaussian_nb(X_train, y_train):
    logger.info("Training GaussianNB (no tuning)")
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    return nb

def tune_catboost(X_train, y_train, X_val, y_val):
    logger.info("Tuning CatBoost with early stopping")
    pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
    cb = CatBoostClassifier(
        verbose=0, random_state=42, class_weights=[1, pos_weight]
    )
    param_grid = {
        'iterations': [100, 200, 300],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        cb, param_grid, n_iter=20, scoring='f1', cv=cv, n_jobs=-1, random_state=42
    )
    search.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=20)
    logger.info(f"Best CatBoost params: {search.best_params_}")
    return search.best_estimator_

def train_voting_ensemble(models: dict, X_train, y_train):
    logger.info("Building Voting Ensemble")
    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        voting='soft',
        n_jobs=-1
    )
    ensemble.fit(X_train, y_train)
    return ensemble

# ---------------------- Main Runner ---------------------- #

def train_models(X_train, y_train, X_val, y_val):
    models = {}

    models['lr'] = tune_logistic_regression(X_train, y_train)
    models['rf'] = tune_random_forest(X_train, y_train)
    models['nb'] = train_gaussian_nb(X_train, y_train)
    models['cb'] = tune_catboost(X_train, y_train, X_val, y_val)

    models['ensemble'] = train_voting_ensemble(models, X_train, y_train)

    return models

# ---------------------- Model Evaluation ---------------------- #

def evaluate_model(name: str, model, X_test: pd.DataFrame, y_test: pd.Series):
    logger.info(f"Evaluating {name}")

    # Predict probabilities if available
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        logger.info(f"{name} F1 score (default threshold 0.5): {f1:.3f}")
        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, zero_division=0))
        return

    # Tune threshold
    best_thresh = 0.5
    best_f1 = 0
    for thresh in np.arange(0.2, 0.81, 0.05):
        preds = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_test, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    logger.info(f"{name} best threshold: {best_thresh:.2f}, F1: {best_f1:.3f}")

    final_preds = (y_proba >= best_thresh).astype(int)
    report = classification_report(y_test, final_preds, output_dict=True, zero_division=0)
    print(f"\n--- {name} ---")
    print(classification_report(y_test, final_preds, zero_division=0))

    # Save metrics
    os.makedirs("output", exist_ok=True)
    model_metrics_file = "output/models_metrics.json"
    if os.path.exists(model_metrics_file):
        with open(model_metrics_file, "r") as f:
            metrics_all = json.load(f)
    else:
        metrics_all = {}

    metrics_all[name] = {
        "f1": best_f1,
        "precision": report['1']['precision'],
        "recall": report['1']['recall'],
        "best_threshold": best_thresh
    }

    with open(model_metrics_file, "w") as f:
        json.dump(metrics_all, f, indent=4)

    # Save feature importance
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
    elif hasattr(model, 'coef_'):
        imp = model.coef_.ravel()
    else:
        logger.info(f"No feature importances for model {name}")
        return

    feature_importances = {feat: float(val) for feat, val in zip(X_test.columns, imp)}
    features_sorted = dict(sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True))

    with open(f"output/features_importance_{name}.json", "w") as f:
        json.dump(features_sorted, f, indent=4)