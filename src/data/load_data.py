import pandas as pd
def load_data_and_label(assess_fp: str, labels_fp: str, stage: int):
    df_ass = pd.read_csv(assess_fp)
    df_lab = pd.read_csv(labels_fp)

    # Pick relevant course stage
    stage_name = "seamanship_command_substage" if stage == 1 else "fundamental_stage"
    df_lab_stage = df_lab[df_lab['course_stage'] == stage_name]
    df_lab_stage = df_lab_stage[df_lab_stage['finished'].isin([0.0, 1.0])]

    if stage == 2:
        df_lab_s1 = df_lab[df_lab['course_stage'] == 'seamanship_command_substage']
        df_lab_s1 = df_lab_s1[df_lab_s1['finished'] == 1.0]
        df_lab_stage = df_lab_stage[df_lab_stage['id'].isin(df_lab_s1['id'])]

    df = df_ass.merge(df_lab_stage[['id', 'finished', 'score']], on='id', how='inner')
    df = df.rename(columns={'finished': 'target', 'score': 'label_score'})
    df['target'] = df['target'].astype(int)

    return df