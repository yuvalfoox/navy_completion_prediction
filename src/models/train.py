import logging
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, classification_report

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

def prepare_features(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    logger.info("Imputing missing values")
    imp = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

    logger.info("Filtering features with >30% zeros and low variance")
    mask = (X_imp == 0).mean() < 0.3
    X_filt = X_imp.loc[:, mask]
    vt = VarianceThreshold(threshold=0.01)
    X_vt = pd.DataFrame(vt.fit_transform(X_filt), columns=X_filt.columns[vt.get_support()])

    logger.info("Selecting features via mutual information")
    mi = mutual_info_classif(X_vt, y, random_state=42)
    sel = mi > 0.001
    X_sel = X_vt.iloc[:, sel]
    logger.info(f"Selected {X_sel.shape[1]} features")
    return X_sel

def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Logistic Regression
    pipe_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(solver='lbfgs', max_iter=2000))
    ])
    lr_params = {'clf__C': [0.01, 0.1, 1], 'clf__class_weight': [None, 'balanced']}
    logger.info("Tuning Logistic Regression")
    lr_gs = GridSearchCV(pipe_lr, lr_params, scoring='f1', cv=cv, n_jobs=-1)
    lr_gs.fit(X_train, y_train)
    best_lr = lr_gs.best_estimator_
    logger.info(f"LR best params: {lr_gs.best_params_}")

    # Random Forest
    rf_params = {'n_estimators': [100], 'max_depth': [None, 10], 'class_weight': [None, 'balanced']}
    logger.info("Tuning Random Forest")
    rf_gs = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, scoring='f1', cv=cv, n_jobs=-1)
    rf_gs.fit(X_train, y_train)
    best_rf = rf_gs.best_estimator_
    logger.info(f"RF best params: {rf_gs.best_params_}")

    # XGBoost
    logger.info("Training XGBoost")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)

    # LightGBM
    logger.info("Training LightGBM")
    lgb = LGBMClassifier(random_state=42)
    lgb.fit(X_train, y_train)

    # CatBoost
    logger.info("Training CatBoost")
    cb = CatBoostClassifier(verbose=0, random_state=42)
    cb.fit(X_train, y_train)

    # Voting Ensemble
    ensemble = VotingClassifier(
        estimators=[('lr', best_lr), ('rf', best_rf), ('xgb', xgb), ('lgb', lgb), ('cb', cb)],
        voting='soft', n_jobs=-1
    )
    ensemble.fit(X_train, y_train)
    logger.info("Ensemble trained")

    return {'lr': best_lr, 'rf': best_rf, 'xgb': xgb, 'lgb': lgb, 'cb': cb, 'ensemble': ensemble}


def evaluate_model(name: str, model, X_test: pd.DataFrame, y_test: pd.Series):
    logger.info(f"Evaluating {name}")
    X_input = X_test if hasattr(model, 'named_steps') else X_test.values
    y_pred = model.predict(X_input)
    logger.info(f"{name} F1: {f1_score(y_test, y_pred):.3f}")
    print(f"--- {name} ---")
    print(classification_report(y_test, y_pred, zero_division=0))