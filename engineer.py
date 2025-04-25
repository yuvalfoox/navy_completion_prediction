
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['assessment_size'] = df.groupby('main_assessment_date')['main_assessment_date'].transform('count')
    df['day_of_week'] = pd.to_datetime(df['main_assessment_date']).dt.dayofweek
    df['short_assessment'] = (df['day_of_week'] != 6).astype(int)

    for col in ['quality_group_score', 'cognition', 'initial_rating_score']:
        if col in df.columns:
            df[col] = df[col].replace(0, df[df[col] > 0][col].median())

    test_cols = [col for col in df.columns if '_test' in col]
    df['tests_avg'] = df[test_cols].mean(axis=1)
    df['assessment_avg'] = df.groupby('main_assessment_date')['tests_avg'].transform('mean')
    global_avg = df['tests_avg'].mean()
    df['adjusted_tests_avg'] = df['tests_avg'] * (global_avg / df['assessment_avg'])

    df['gender'] = df['gender'].map({'זכר': 1, 'נקבה': 0}).fillna(0)
    df['stem_units'] = pd.Categorical(df['stem_units']).codes

    drop_cols = ['id', 'trainee_number', 'Source.Name', 'completion_method',
                 'conduct_two_assessments', 'main_assessment_date', 'tests_avg', 'assessment_avg']
    return df.drop(columns=[col for col in drop_cols if col in df.columns])
