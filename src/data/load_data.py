import pandas as pd
import numpy as np

# Map stage number to label file's course_stage
_STAGE_MAP = {
    1: 'fundamental_stage',
    2: 'seamanship_command_substage'
}


def extract_date(source_name):
    try:
        return pd.to_datetime(
            '.'.join(str(source_name).split('.')[:3]),
            format='%d.%m.%y'
        )
    except:
        return None


def load_and_label(
        data_filepath: str,
        stage: int,
        labels_filepath: str = 'src/data/input/labels.csv'
) -> pd.DataFrame:
    """
    Load the CSV version of assessments and merge with labels.
    Args:
      data_filepath: path to navy_assessments_v1.csv
      stage:         1 or 2
      labels_filepath: path to labels.csv
    Returns:
      DataFrame with features + 'target' + 'final_instructor_score'
    """
    # 1. Load raw assessments (CSV)
    df = pd.read_csv(data_filepath)
    df['main_assessment_date'] = df['Source.Name'].apply(extract_date)
    df = df.dropna(subset=['main_assessment_date'])
    df = df.sort_values('main_assessment_date')
    df['assessment_counter'] = df.groupby('main_assessment_date').ngroup() + 1

    # 2. Load labels and filter by stage
    labels = pd.read_csv(labels_filepath)
    target_stage = _STAGE_MAP.get(stage)
    if target_stage is None:
        raise ValueError(f"Stage must be 1 or 2, got {stage}")
    lab = labels[labels['course_stage'] == target_stage].copy()

    # 3. Binarize finished flag
    lab['finished_bin'] = np.where(lab['finished'] == 1.0, 1, 0)

    # 4. Merge into main DataFrame
    df = df.merge(
        lab[['id', 'finished_bin', 'score']],
        on='id',
        how='left'
    ).rename(
        columns={'finished_bin': 'target', 'score': 'final_instructor_score'}
    )

    # 5. Drop unlabeled rows
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)

    return df
