import pandas as pd
import numpy as np
from src.features.process_test_scores import process_test_scores
from src.config import zero_threshold, var_threshold, mi_threshold


def preprocess_assessment_data(df: pd.DataFrame, date_column: str = 'Source.Name') -> pd.DataFrame:
    """
    Preprocess assessments: extract dates, sort, create counters, additional features.
    """
    def extract_date(filename):
        try:
            date_parts = filename.split('.')[:3]
            date_string = '.'.join(date_parts)
            return pd.to_datetime(date_string, format='%d.%m.%y')
        except (ValueError, AttributeError):
            return None

    # Extract and fix dates
    df['main_assessment_date'] = df[date_column].apply(extract_date)

    # Drop invalid dates and columns
    df = df.drop(columns=[date_column], errors='ignore')
    df = df.dropna(subset=['main_assessment_date'])

    # Sort by date
    df = df.sort_values('main_assessment_date')

    # Create assessment counter
    df['assessment_counter'] = df.groupby('main_assessment_date').ngroup() + 1

    # Day of week
    df['day_of_week'] = df['main_assessment_date'].dt.dayofweek
    df['short_assessment'] = (df['day_of_week'] != 6).astype(int)

    # Assessment size
    df['assessment_size'] = df.groupby('main_assessment_date')['main_assessment_date'].transform('count')

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering after data is loaded.
    """
    df = df.copy()

    # Basic engineered columns
    if 'main_assessment_date' not in df.columns and 'Source.Name' in df.columns:
        df = preprocess_assessment_data(df, date_column='Source.Name')

    df = process_test_scores(df)

    if 'stem_units' in df.columns:
        df['stem_units'] = pd.Categorical(df['stem_units']).codes

    if 'conduct_two_assessments' in df.columns:
        df['conduct_two_assessments'] = df['conduct_two_assessments'].notna().astype(int)

    if 'final_instructor_score' in df.columns and 'adjusted_tests_avg' in df.columns:
        df['instructor_score_ratio'] = df['final_instructor_score'] / (df['adjusted_tests_avg'] + 1e-6)

    if 'teamwork_final' in df.columns and 'stress_resilience_final' in df.columns:
        df['teamwork_stress_interaction'] = df['teamwork_final'] * df['stress_resilience_final']

    if 'gps_test' in df.columns:
        df['log_gps_test'] = np.log1p(df['gps_test'])

    if 'land_command' in df.columns and 'sea_command' in df.columns:
        df['land_sea_command_ratio'] = df['land_command'] / (df['sea_command'] + 1e-6)

    df = df.drop(columns=['id', 'trainee_number', 'Source.Name', 'completion_method'], errors='ignore')

    return df
