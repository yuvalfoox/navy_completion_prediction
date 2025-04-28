import pandas as pd

def load_data_and_label(assess_fp: str, labels_fp: str, stage: int):
    df_ass = pd.read_csv(assess_fp)
    df_lab = pd.read_csv(labels_fp)

    if stage == 1:
        df_lab_stage = df_lab[df_lab['course_stage'] == 'seamanship_command_substage']
        df_lab_stage = df_lab_stage[df_lab_stage['finished'].isin([0.0, 1.0])]

    elif stage == 2:
        df_lab_s1 = df_lab[df_lab['course_stage'] == 'seamanship_command_substage']
        df_lab_s1 = df_lab_s1[df_lab_s1['finished'].isin([0.0, 1.0])]
        df_lab_s1 = df_lab_s1[['id', 'finished']].rename(columns={'finished': 'stage1_finished'})

        df_lab_s2 = df_lab[df_lab['course_stage'] == 'fundamental_stage']
        df_lab_s2 = df_lab_s2[df_lab_s2['finished'].isin([0.0, 1.0])]
        df_lab_s2 = df_lab_s2[['id', 'finished', 'score']]

        merged = pd.merge(df_lab_s1, df_lab_s2, how='outer', on='id', indicator=True)

        merged['final_stage2_target'] = merged.apply(
            lambda row: 0 if row['stage1_finished'] == 0 else row['finished'], axis=1
        )

        df_lab_stage = merged[['id', 'final_stage2_target', 'score']].rename(columns={
            'final_stage2_target': 'finished',
            'score': 'label_score'
        })
        df_lab_stage = df_lab_stage.dropna(subset=['finished'])
        df_lab_stage['finished'] = df_lab_stage['finished'].astype(int)

    else:
        raise ValueError(f"Invalid stage number {stage}")

    df = df_ass.merge(df_lab_stage[['id', 'finished']], on='id', how='inner')
    df = df.rename(columns={'finished': 'target'})
    df['target'] = df['target'].astype(int)

    return df
