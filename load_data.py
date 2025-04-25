
import pandas as pd
import numpy as np

def extract_date(source_name):
    try:
        return pd.to_datetime('.'.join(str(source_name).split('.')[:3]), format='%d.%m.%y')
    except:
        return None

def load_and_label(filepath: str, stage: int):
    df = pd.read_excel(filepath)
    df['main_assessment_date'] = df['Source.Name'].apply(extract_date)
    df = df.dropna(subset=['main_assessment_date'])
    df = df.sort_values('main_assessment_date')
    df['assessment_counter'] = df.groupby('main_assessment_date').ngroup() + 1

    stage2_ids = set(df.loc[df['conduct_two_assessments'] == 2.0, 'trainee_number'])

    if stage == 1:
        def label_stage1(row):
            if row['trainee_number'] in stage2_ids:
                return 1
            if pd.isna(row['completion_method']):
                return np.nan
            return 1 if row['completion_method'] == 'completed' else 0
        df['target'] = df.apply(label_stage1, axis=1)
        df = df[df['conduct_two_assessments'].isna()]
    elif stage == 2:
        df = df[df['conduct_two_assessments'] == 2.0]
        df['target'] = df['completion_method'].apply(
            lambda x: 1 if x == 'completed' else (0 if x in ['לא סיים','ויתר','נפסל רפואית'] else np.nan)
        )
    else:
        raise ValueError("Stage must be 1 or 2")

    df = df.dropna(subset=['target'])
    return df
