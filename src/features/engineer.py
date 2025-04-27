import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the assessment DataFrame:
      - Drop non-predictive and non-numeric columns
      - Create numeric count and flag features
      - Aggregate domain and overall scores
      - One-hot encode categorical fields
    """
    df = df.copy()

    # Drop columns that are not predictive or non-numeric
    cols_to_drop = ['Source.Name', 'assessment_date', 'completion_method', 'conduct_two_assessments']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    # One-hot encode gender if present
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'זכר': 'male', 'נקבה': 'female'}).fillna('unknown')
        dummies = pd.get_dummies(df['gender'], prefix='gender')
        df = pd.concat([df, dummies], axis=1)
        df.drop(columns=['gender'], inplace=True)

    # Count of assessment attempts per trainee (requires id temporarily)
    if 'id' in df.columns:
        attempt_counts = df.groupby('id').size().to_dict()
        df['attempt_count'] = df['id'].map(attempt_counts)
    else:
        df['attempt_count'] = 1

    # Flags if two-part assessment occurred
    if 'conduct_two_assessments' in df.columns:
        df['two_part_assessment'] = df['conduct_two_assessments'].fillna(0).astype(int)

    # Aggregate machinery and electronics scores
    mech_cols = [c for c in df.columns if c.startswith('machinery_')]
    elec_cols = [c for c in df.columns if c.startswith('electronics_')]
    if mech_cols:
        df['total_mechanical_score'] = df[mech_cols].sum(axis=1)
    if elec_cols:
        df['total_electrical_score'] = df[elec_cols].sum(axis=1)
    if mech_cols and elec_cols:
        df['tech_diff'] = df['total_mechanical_score'] - df['total_electrical_score']

    # Sum all *_score fields for overall assessment performance
    score_cols = [c for c in df.columns if c.endswith('_score')]
    if score_cols:
        df['assessment_sum_score'] = df[score_cols].sum(axis=1)

    # Finally drop id
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

    return df