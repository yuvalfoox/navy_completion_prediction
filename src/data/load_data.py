import pandas as pd
import numpy as np

# Map stage number to label file's course_stage
_STAGE_MAP = {
    1: 'fundamental_stage',
    2: 'seamanship_command_substage'
}

def extract_date(source_name):
    try:
        parts = str(source_name).split('.')[:3]
        return pd.to_datetime('.'.join(parts), format='%d.%m.%y')
    except Exception:
        return pd.NaT


def load_and_label(
        data_filepath: str,
        stage: int,
        labels_filepath: str = 'labels.csv'
) -> pd.DataFrame:
    """
    Load assessment data and attach stage-specific labels:
    - Stage 1: include only those with fundamental_stage labels
    - Stage 2: include Stage1 failures (auto-negative) and real Stage2 labels

    Returns DataFrame with one record per trainee, including 'target' and 'final_instructor_score'
    """
    # 1. Load raw data and parse dates
    df = pd.read_csv(data_filepath)
    df['assessment_date'] = df['Source.Name'].apply(extract_date)
    df = df.dropna(subset=['assessment_date']).copy()
    # Keep latest record per trainee
    df = df.sort_values(['id', 'assessment_date']).drop_duplicates(subset='id', keep='last')

    # 2. Load labels and build lookup dicts
    labels = pd.read_csv(labels_filepath)
    lab1 = labels[labels['course_stage'] == _STAGE_MAP[1]]
    lab2 = labels[labels['course_stage'] == _STAGE_MAP[2]]
    lab1_finished = lab1.set_index('id')['finished'].to_dict()
    lab1_score = lab1.set_index('id')['score'].to_dict()
    lab2_finished = lab2.set_index('id')['finished'].to_dict()
    lab2_score = lab2.set_index('id')['score'].to_dict()

    if stage == 1:
        # Stage1: filter and map targets/scores
        df1 = df[df['id'].isin(lab1_finished)].copy()
        df1['target'] = df1['id'].apply(lambda i: 1 if lab1_finished.get(i) == 1.0 else 0)
        df1['final_instructor_score'] = df1['id'].apply(lambda i: lab1_score.get(i, np.nan))
        return df1.reset_index(drop=True)

    # Stage2: include Stage1 failures and Stage2 labels
    ids_fail1 = [i for i, f in lab1_finished.items() if f != 1.0]
    ids2 = list(lab2_finished.keys())
    include_ids = set(ids_fail1) | set(ids2)
    df2 = df[df['id'].isin(include_ids)].copy()

    # Map Stage2 target: passed Stage1 AND finished Stage2
    def map_stage2(i):
        if lab1_finished.get(i) != 1.0:
            return 0
        return 1 if lab2_finished.get(i) == 1.0 else 0
    df2['target'] = df2['id'].apply(map_stage2)
    df2['final_instructor_score'] = df2['id'].apply(lambda i: lab2_score.get(i, np.nan))
    return df2.reset_index(drop=True)