import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Assessment-level statistics
    df['assessment_size'] = df.groupby('main_assessment_date')['main_assessment_date'].transform('count')
    df['day_of_week'] = pd.to_datetime(df['main_assessment_date']).dt.dayofweek
    df['short_assessment'] = (df['day_of_week'] != 6).astype(int)

    # Replace zeros in key scores with median
    for col in ['quality_group_score', 'cognition', 'initial_rating_score']:
        if col in df.columns:
            med = df.loc[df[col] > 0, col].median()
            df[col] = df[col].replace(0, med)

    # Test average adjustments
    test_cols = [c for c in df.columns if '_test' in c]
    df['tests_avg'] = df[test_cols].mean(axis=1)
    df['assessment_avg'] = df.groupby('main_assessment_date')['tests_avg'].transform('mean')
    global_avg = df['tests_avg'].mean()
    df['adjusted_tests_avg'] = df['tests_avg'] * (global_avg / df['assessment_avg'])

    # Categorical encodings
    df['gender'] = df['gender'].map({'זכר': 1, 'נקבה': 0}).fillna(0)
    df['stem_units'] = pd.Categorical(df['stem_units']).codes

    # Drop raw and intermediate columns
    drop_cols = [
        'id', 'trainee_number', 'Source.Name', 'completion_method',
        'conduct_two_assessments', 'main_assessment_date',
        'tests_avg', 'assessment_avg'
    ]
    return df.drop(columns=[c for c in drop_cols if c in df.columns])
