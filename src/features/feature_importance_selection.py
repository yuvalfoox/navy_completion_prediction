import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_class_weight
from typing import List, Literal

def feature_importance_selection(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_list: List[str],
        target: str,
        model_type: Literal['rf', 'lr', 'cb'] = 'rf',
        feature_selection_threshold: float = 0.001,
        n_estimators: int = 100,
        n_repeats: int = 50,
        random_state: int = 42,
        n_jobs: int = -1
) -> List[str]:
    """
    Select important features using permutation importance with different models.

    Args:
    - train_df (pd.DataFrame): Training data
    - val_df (pd.DataFrame): Validation data
    - feature_list (List[str]): Feature columns
    - target (str): Target column
    - model_type (str): Which model to use: 'rf' (Random Forest), 'lr' (Logistic Regression), 'cb' (CatBoost)
    - feature_selection_threshold (float): Minimum importance threshold
    - n_estimators (int): Number of trees (for RF or CB)
    - n_repeats (int): Permutation repeats
    - random_state (int): Random seed
    - n_jobs (int): Parallelism

    Returns:
    - List[str]: Selected features
    """

    # --- Validation ---
    if not all(col in train_df.columns for col in feature_list + [target]):
        raise ValueError("Some specified features or target missing from training dataframe.")
    if not all(col in val_df.columns for col in feature_list + [target]):
        raise ValueError("Some specified features or target missing from validation dataframe.")

    X_train = train_df[feature_list]
    y_train = train_df[target]

    X_val = val_df[feature_list]
    y_val = val_df[target]

    print(f"Data loaded: train {X_train.shape}, val {X_val.shape}")

    # --- Class Weights ---
    class_weight = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight = dict(zip(np.unique(y_train), class_weight))

    # --- Model Initialization ---
    if model_type == 'rf':
        base_model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs
        )
    elif model_type == 'lr':
        base_model = LogisticRegression(
            max_iter=2000,
            class_weight=class_weight,
            random_state=random_state,
            solver='saga'
        )
    elif model_type == 'cb':
        base_model = CatBoostClassifier(
            iterations=n_estimators,
            random_state=random_state,
            class_weights=[class_weight[0], class_weight[1]],
            verbose=0
        )
    else:
        raise ValueError("Unsupported model_type. Choose from: 'rf', 'lr', 'cb'.")

    base_model.fit(X_train, y_train)
    print(f"Model ({model_type}) trained.")

    # --- Permutation Importance ---
    result = permutation_importance(
        base_model,
        X_val,
        y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs
    )
    print("Permutation importance calculated.")

    # --- Feature Importance Dataframe ---
    feature_importances = pd.DataFrame({
        'feature': feature_list,
        'importance': result.importances_mean
    }).sort_values(by='importance', ascending=False)

    # --- Feature Selection ---
    selected_features = feature_importances[feature_importances.importance >= feature_selection_threshold]['feature'].tolist()
    print(f"Selected {len(selected_features)} features out of {len(feature_list)}.")

    return selected_features
