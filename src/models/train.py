import logging
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
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

    logger.info("Filtering features with >30% zeros")
    mask = (X_imp == 0).mean() < 0.3
    X_filt = X_imp.loc[:, mask]

    logger.info("Applying low-variance filter")
    vt = VarianceThreshold(threshold=0.01)
    X_vt = pd.DataFrame(vt.fit_transform(X_filt), columns=X_filt.columns[vt.get_support()])

    logger.info("Selecting features via mutual information")
    mi = mutual_info_classif(X_vt, y, random_state=42)
    sel = mi > 0.001
    X_sel = X_vt.iloc[:, sel]
    logger.info(f"Selected {X_sel.shape[1]} features after MI filtering")
    return X_sel

def train_models(X_train: pd.DataFrame, y_train: pd.Series):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Logistic Regression pipeline + tuning
    pipe_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(solver='lbfgs', max_iter=2000))
    ])
    lr_params = {
        'clf__C': [0.01, 0.1, 1, 10],
        'clf__class_weight': [None, 'balanced']
    }
    logger.info("Tuning Logistic Regression")
    lr_gs = GridSearchCV(pipe_lr, lr_params, scoring='f1', cv=cv, n_jobs=-1)
    lr_gs.fit(X_train, y_train)
    best_lr = lr_gs.best_estimator_
    logger.info(f"LR best params: {lr_gs.best_params_} → F1={lr_gs.best_score_:.3f}")

    # Random Forest tuning
    rf_params = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'class_weight': [None, 'balanced']
    }
    logger.info("Tuning Random Forest")
    rf_gs = GridSearchCV(RandomForestClassifier(random_state=42),
                         rf_params, scoring='f1', cv=cv, n_jobs=-1)
    rf_gs.fit(X_train, y_train)
    best_rf = rf_gs.best_estimator_
    logger.info(f"RF best params: {rf_gs.best_params_} → F1={rf_gs.best_score_:.3f}")

    # GaussianNB tuning
    nb_params = {'var_smoothing': [1e-9, 1e-8, 1e-7]}
    logger.info("Tuning GaussianNB")
    nb_gs = GridSearchCV(GaussianNB(), nb_params, scoring='f1', cv=cv, n_jobs=-1)
    nb_gs.fit(X_train, y_train)
    best_nb = nb_gs.best_estimator_
    logger.info(f"NB best params: {nb_gs.best_params_} → F1={nb_gs.best_score_:.3f}")

    # CatBoost (no CV)
    logger.info("Training CatBoostClassifier")
    cb = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        verbose=0,
        random_state=42,
        class_weights=[1.0, len(y_train)/y_train.sum()]
    )
    cb.fit(X_train, y_train)
    logger.info("CatBoost trained")

    # Voting Ensemble
    logger.info("Building VotingClassifier ensemble")
    ensemble = VotingClassifier(
        estimators=[('lr', best_lr), ('rf', best_rf),
                    ('nb', best_nb), ('cb', cb)],
        voting='soft',
        n_jobs=-1
    )
    ensemble.fit(X_train, y_train)
    logger.info("Ensemble trained")

    return {
        'lr': best_lr,
        'rf': best_rf,
        'nb': best_nb,
        'cb': cb,
        'ensemble': ensemble
    }

def evaluate_model(name: str, model, X_test: pd.DataFrame, y_test: pd.Series):
    logger.info(f"Evaluating {name}")
    # For pipeline models, keep DataFrame; else use NumPy array
    X_input = X_test if hasattr(model, 'named_steps') else X_test.values
    y_pred = model.predict(X_input)
    f1 = f1_score(y_test, y_pred)
    logger.info(f"{name} test F1: {f1:.3f}")

    print(f"--- {name} ---")
    print(classification_report(y_test, y_pred, zero_division=0))
