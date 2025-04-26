import logging
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score

# configure module-level logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

def prepare_features(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    logger.info(f"Preparing features: imputing missing values")
    imp = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

    logger.info(f"Filtering features with >30% zeros")
    keep = (X_imp == 0).mean() < 0.3
    X_filt = X_imp.loc[:, keep]

    logger.info(f"Applying low-variance filter")
    vt = VarianceThreshold(threshold=0.01)
    X_vt = pd.DataFrame(vt.fit_transform(X_filt),
                        columns=X_filt.columns[vt.get_support()])

    logger.info(f"Computing mutual information for feature selection")
    mi = mutual_info_classif(X_vt, y, random_state=42)
    keep_mi = mi > 0.001
    X_sel = X_vt.iloc[:, keep_mi]
    logger.info(f"Selected {X_sel.shape[1]} features after MI filtering")
    return X_sel

def train_models(X_train: pd.DataFrame, y_train: pd.Series):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Logistic Regression tuning
    logger.info("Tuning Logistic Regression")
    lr_grid = {
        'C': [0.01, 0.1, 1, 10],
        'class_weight': [None, 'balanced']
    }
    lr = GridSearchCV(
        LogisticRegression(max_iter=500, solver='lbfgs'),
        lr_grid, scoring='f1', cv=cv, n_jobs=-1
    )
    lr.fit(X_train, y_train)
    logger.info(f"Best LR params: {lr.best_params_} → F1={lr.best_score_:.3f}")
    best_lr = lr.best_estimator_

    # Random Forest tuning
    logger.info("Tuning Random Forest")
    rf_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'class_weight': [None, 'balanced']
    }
    rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_grid, scoring='f1', cv=cv, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    logger.info(f"Best RF params: {rf.best_params_} → F1={rf.best_score_:.3f}")
    best_rf = rf.best_estimator_

    # GaussianNB tuning
    logger.info("Tuning GaussianNB")
    nb_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7]}
    nb = GridSearchCV(GaussianNB(), nb_grid, scoring='f1', cv=cv, n_jobs=-1)
    nb.fit(X_train, y_train)
    logger.info(f"Best NB params: {nb.best_params_} → F1={nb.best_score_:.3f}")
    best_nb = nb.best_estimator_

    # CatBoost (default params, fast)
    logger.info("Training CatBoostClassifier (no CV)")
    cb = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        verbose=0,
        random_state=42,
        class_weights=[1.0, (len(y_train)/(y_train.sum()))]  # balance positive class
    )
    cb.fit(X_train, y_train)
    logger.info("CatBoost trained.")

    # Ensemble
    logger.info("Building soft-voting ensemble")
    ensemble = VotingClassifier(
        estimators=[
            ('lr', best_lr),
            ('rf', best_rf),
            ('nb', best_nb),
            ('cb', cb)
        ],
        voting='soft',
        n_jobs=-1
    )
    ensemble.fit(X_train, y_train)
    logger.info("Ensemble trained.")

    return {
        'lr': best_lr,
        'rf': best_rf,
        'nb': best_nb,
        'cb': cb,
        'ensemble': ensemble
    }

def evaluate_model(name: str, model, X_test: pd.DataFrame, y_test: pd.Series):
    logger.info(f"Evaluating {name}")
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    logger.info(f"{name} F1 on test: {f1:.3f}")
    print(f"--- {name} ---")
    print(classification_report(y_test, y_pred, zero_division=0))
