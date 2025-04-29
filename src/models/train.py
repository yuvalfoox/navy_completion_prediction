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
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_curve, precision_recall_curve

from src.config import (
    var_threshold, mi_threshold, zero_threshold,
    random_state, cv_folds, random_search_iter
)

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

    logger.info(f"Filtering features with >{zero_threshold}% zeros")
    zero_frac = (X_train_imp == 0).mean()
    keep_mask = zero_frac < (zero_threshold / 100)
    X_filt = X_train_imp.loc[:, keep_mask]

    logger.info(f"Applying low-variance filter (threshold={var_threshold})")
    vt = VarianceThreshold(threshold=var_threshold)
    X_vt = pd.DataFrame(vt.fit_transform(X_filt), columns=X_filt.columns[vt.get_support()])

    logger.info(f"Computing mutual information filtering (mi_threshold={mi_threshold})")
    mi = mutual_info_classif(X_vt, y_train, random_state=random_state)
    keep_mi = mi > mi_threshold
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
        ('clf', LogisticRegression(max_iter=2000, solver='saga', random_state=random_state, class_weight='balanced'))
    ])
    param_grid = {
        'clf__C': np.logspace(-3, 2, 15),
        'clf__penalty': ['l1', 'l2'],
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        lr_pipe, param_grid, n_iter=random_search_iter, scoring='f1', cv=cv, n_jobs=-1, random_state=random_state
    )
    search.fit(X_train, y_train)
    logger.info(f"Best LR params: {search.best_params_}")
    return search.best_estimator_

def tune_random_forest(X_train, y_train):
    logger.info("Tuning Random Forest")
    rf = RandomForestClassifier(random_state=random_state, class_weight='balanced')
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [5, 10, 20, None],
        'min_samples_leaf': [1, 2, 5]
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        rf, param_grid, n_iter=random_search_iter, scoring='f1', cv=cv, n_jobs=-1, random_state=random_state
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
    cb = CatBoostClassifier(verbose=0, random_state=random_state, class_weights=[1, pos_weight])

    param_grid = {
        'iterations': [100, 200, 300],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1]
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        cb, param_grid, n_iter=random_search_iter, scoring='f1', cv=cv, n_jobs=-1, random_state=random_state
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

def evaluate_model(name: str, model, X_test: pd.DataFrame, y_test: pd.Series, output_dir: str):
    logger.info(f"Evaluating {name}")

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        logger.info(f"{name} F1 score (default threshold 0.5): {f1:.3f}")
        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, zero_division=0))
        return

    best_thresh = 0.5
    best_f1 = 0
    threshold_scores = []

    for thresh in np.arange(0.2, 0.81, 0.02):
        preds = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_test, preds)
        threshold_scores.append((thresh, f1))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    logger.info(f"{name} best threshold: {best_thresh:.2f}, F1: {best_f1:.3f}")
    final_preds = (y_proba >= best_thresh).astype(int)

    report = classification_report(y_test, final_preds, output_dict=True, zero_division=0)

    os.makedirs(output_dir, exist_ok=True)
    model_metrics_file = os.path.join(output_dir, "models_metrics.json")

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

    cm = confusion_matrix(y_test, final_preds).tolist()
    with open(os.path.join(output_dir, f"confusion_matrix_{name}.json"), "w") as f:
        json.dump(cm, f, indent=4)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    with open(os.path.join(output_dir, f"roc_curve_{name}.json"), "w") as f:
        json.dump(roc_data, f, indent=4)

    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    prc_data = {"precision": prec.tolist(), "recall": rec.tolist()}
    with open(os.path.join(output_dir, f"prc_curve_{name}.json"), "w") as f:
        json.dump(prc_data, f, indent=4)

    thresh_scores = {str(t): float(s) for t, s in threshold_scores}
    with open(os.path.join(output_dir, f"threshold_scores_{name}.json"), "w") as f:
        json.dump(thresh_scores, f, indent=4)

    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
    elif hasattr(model, 'coef_'):
        imp = model.coef_.ravel()
    else:
        logger.info(f"No feature importances for model {name}")
        return

    feature_importances = {feat: float(val) for feat, val in zip(X_test.columns, imp)}
    features_sorted = dict(sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True))

    with open(os.path.join(output_dir, f"features_importance_{name}.json"), "w") as f:
        json.dump(features_sorted, f, indent=4)
