import pandas as pd
import numpy as np

# Map our stage argument to the label file's course_stage
_STAGE_MAP = {
    1: 'fundamental_stage',
    2: 'seamanship_command_substage'
}

def extract_date(source_name):
    try:
        return pd.to_datetime('.'.join(str(source_name).split('.')[:3]),
                              format='%d.%m.%y')
    except:
        return None

def load_and_label(data_filepath: str,
                   stage: int,
                   labels_filepath: str = 'navy_assessment.xlsx - labels.csv'):
    # 1. Load raw assessment data
    df = pd.read_excel(data_filepath)
    df['main_assessment_date'] = df['Source.Name'].apply(extract_date)
    df = df.dropna(subset=['main_assessment_date'])
    df = df.sort_values('main_assessment_date')
    df['assessment_counter'] = df.groupby('main_assessment_date').ngroup() + 1

    # 2. Load and filter labels
    labels = pd.read_csv(labels_filepath)
    target_stage = _STAGE_MAP.get(stage)
    if target_stage is None:
        raise ValueError(f"Stage must be 1 or 2, got {stage}")
    lab = labels[labels['course_stage'] == target_stage].copy()

    # Convert finished to binary: only 1.0 → 1, else 0
    lab['finished_bin'] = np.where(lab['finished'] == 1.0, 1, 0)

    # 3. Merge labels into main dataframe on 'id'
    df = df.merge(
        lab[['id', 'finished_bin', 'score']],
        on='id',
        how='left'
    ).rename(columns={'finished_bin': 'target',
                      'score': 'final_instructor_score'})

    # 4. Drop any rows without a label
    df = df.dropna(subset=['target'])

    return df
