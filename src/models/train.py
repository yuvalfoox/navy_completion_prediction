import logging
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def prepare_features(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    logger.info("Preparing features: imputing missing values")
    imp = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

    logger.info("Filtering features with >30% zeros")
    zero_frac = (X_imp == 0).mean()
    keep_mask = zero_frac < 0.30
    X_filt = X_imp.loc[:, keep_mask]

    logger.info("Applying low-variance filter")
    vt = VarianceThreshold(threshold=0.01)
    X_vt = pd.DataFrame(vt.fit_transform(X_filt), columns=X_filt.columns[vt.get_support()])

    logger.info("Computing mutual information for feature selection")
    mi = mutual_info_classif(X_vt, y, random_state=42)
    keep_mi = mi > 0.001
    X_sel = X_vt.loc[:, keep_mi]

    logger.info(f"Selected {X_sel.shape[1]} features after MI filtering")
    return X_sel


def train_models(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
    models = {}

    # Logistic Regression
    logger.info("Training Logistic Regression")
    lr = LogisticRegression(max_iter=2000, solver='saga', random_state=42)
    lr.fit(X_train, y_train)
    models['lr'] = lr

    # Random Forest
    logger.info("Training Random Forest")
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
    rf.fit(X_train, y_train)
    models['rf'] = rf

    # GaussianNB
    logger.info("Training GaussianNB")
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    models['nb'] = nb

    # CatBoost
    logger.info("Training CatBoost")
    cb = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, verbose=0, random_state=42)
    cb.fit(X_train, y_train)
    models['cb'] = cb

    # Voting ensemble
    logger.info("Building soft voting ensemble")
    ensemble = VotingClassifier(
        estimators=[('lr', models['lr']), ('rf', models['rf']), ('nb', models['nb']), ('cb', models['cb'])],
        voting='soft', n_jobs=-1
    )
    ensemble.fit(X_train, y_train)
    models['ensemble'] = ensemble

    return models


def evaluate_model(name: str, model, X_test: pd.DataFrame, y_test: pd.Series):
    logger.info(f"Evaluating {name}")
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    logger.info(f"{name} F1 score: {f1:.3f}")
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Feature importances
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
    elif hasattr(model, 'coef_'):
        imp = model.coef_.ravel()
    else:
        logger.info(f"No feature importances for model {name}")
        return

    features = X_test.columns
    for feat, val in sorted(zip(features, imp), key=lambda x: abs(x[1]), reverse=True):
        print(f"{feat}: {val:.4f}")
