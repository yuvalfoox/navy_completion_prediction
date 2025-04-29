import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from scipy.stats import randint, uniform

from src.config import random_state, cv_folds, random_search_iter

def optimize_models(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Optimize RandomForest, LogisticRegression, and evaluate DummyClassifier baselines.

    Args:
    - X_train (pd.DataFrame): Training features
    - y_train (pd.Series): Training target

    Returns:
    - Best RandomForest model
    - Best LogisticRegression model
    - Best DummyClassifier model
    """

    # Random Forest Search
    print("Tuning Random Forest...")
    rf_param_dist = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(5, 50),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }

    rf_model = RandomForestClassifier(random_state=random_state, n_jobs=-1, class_weight='balanced')

    rf_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=rf_param_dist,
        n_iter=random_search_iter,
        cv=cv_folds,
        scoring='f1',
        n_jobs=-1,
        random_state=random_state
    )
    rf_search.fit(X_train, y_train)

    print("Best Random Forest parameters:", rf_search.best_params_)
    print("Best RF CV F1 Score:", rf_search.best_score_)

    # Logistic Regression Search
    print("\nTuning Logistic Regression...")
    log_param_dist = {
        'C': uniform(0.01, 10),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }

    log_model = LogisticRegression(random_state=random_state, max_iter=2000, class_weight='balanced')

    log_search = RandomizedSearchCV(
        estimator=log_model,
        param_distributions=log_param_dist,
        n_iter=random_search_iter,
        cv=cv_folds,
        scoring='f1',
        n_jobs=-1,
        random_state=random_state
    )
    log_search.fit(X_train, y_train)

    print("Best Logistic Regression parameters:", log_search.best_params_)
    print("Best LR CV F1 Score:", log_search.best_score_)

    # DummyClassifier Evaluation
    print("\nEvaluating DummyClassifier baselines...")
    dummy_strategies = ['stratified', 'most_frequent', 'prior', 'uniform']
    dummy_scores = []

    for strategy in dummy_strategies:
        dummy = DummyClassifier(strategy=strategy, random_state=random_state)
        dummy.fit(X_train, y_train)
        score = dummy.score(X_train, y_train)
        dummy_scores.append((strategy, score))

    best_dummy = max(dummy_scores, key=lambda x: x[1])

    print(f"Best DummyClassifier strategy: {best_dummy[0]} with train score: {best_dummy[1]:.4f}")

    dummy_model = DummyClassifier(strategy=best_dummy[0], random_state=random_state)
    dummy_model.fit(X_train, y_train)

    return rf_search.best_estimator_, log_search.best_estimator_, dummy_model
