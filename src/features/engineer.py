import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if 'main_assessment_date' not in df.columns and 'Source.Name' in df.columns:
        df['main_assessment_date'] = pd.to_datetime(
            df['Source.Name'].astype(str).str.split('_').str[0],
            dayfirst=True,
            errors='coerce'
        )

    df['assessment_size'] = df.groupby('main_assessment_date')['main_assessment_date'].transform('count')
    df['day_of_week'] = df['main_assessment_date'].dt.dayofweek
    df['short_assessment'] = (df['day_of_week'] != 6).astype(int)

    for col in ['quality_group_score', 'cognition', 'initial_rating_score']:
        if col in df.columns:
            nonzero_median = df.loc[df[col] > 0, col].median()
            df[col] = df[col].replace(0, nonzero_median)

    test_cols = [col for col in df.columns if col.endswith('_test')]
    df['tests_avg'] = df[test_cols].mean(axis=1)
    df['assessment_avg'] = df.groupby('main_assessment_date')['tests_avg'].transform('mean')
    global_avg = df['tests_avg'].mean()
    df['adjusted_tests_avg'] = df['tests_avg'] * (global_avg / df['assessment_avg'])

    df['gender'] = df['gender'].map({'זכר': 1, 'נקבה': 0}).fillna(0)
    df['stem_units'] = pd.Categorical(df['stem_units']).codes

    df = df.drop(columns=['id', 'trainee_number', 'Source.Name', 'completion_method'], errors='ignore')

    return df
