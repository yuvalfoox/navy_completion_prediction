import numpy as np
import pandas as pd
from typing import List

def process_test_scores(
        df: pd.DataFrame,
        special_columns: List[str] = ['quality_group_score', 'cognition', 'initial_rating_score'],
        test_suffix: str = '_test'
) -> pd.DataFrame:
    df_processed = df.copy()

    # Replace 0 with median for special columns
    for col in special_columns:
        if col in df_processed.columns:
            median_value = df_processed[df_processed[col] != 0][col].median()
            df_processed[col] = df_processed[col].replace(0, median_value)

    # Identify test columns
    test_columns = [col for col in df_processed.columns if col.endswith(test_suffix)]
    if not test_columns:
        raise ValueError(f"No columns with suffix {test_suffix}")

    df_processed['tests_avg'] = df_processed[test_columns].mean(axis=1)
    overall_avg = df_processed['tests_avg'].mean()
    df_processed['assessment_avg'] = df_processed.groupby('main_assessment_date')['tests_avg'].transform('mean')
    adjustment_factor = overall_avg / df_processed['assessment_avg']
    df_processed['adjusted_tests_avg'] = df_processed['tests_avg'] * adjustment_factor

    df_processed.drop(columns=['tests_avg', 'assessment_avg'], inplace=True)

    return df_processed
