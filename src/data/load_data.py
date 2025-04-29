import pandas as pd
from src.features.engineer import preprocess_assessment_data

def load_data_and_label(assess_fp: str, labels_fp: str, stage: int):
    """
    Load and preprocess assessments and labels.
    
    Args:
    - assess_fp (str): Path to navy assessments CSV
    - labels_fp (str): Path to labels CSV
    - stage (int): 1 or 2
    
    Returns:
    - Merged, cleaned DataFrame ready for model input
    """

    # --- Load Base Data ---
    df_assess = pd.read_csv(assess_fp)
    df_labels = pd.read_csv(labels_fp)

    # Drop unnecessary fields from base data
    df_assess = df_assess.drop(columns=[
        'initial_rating_score', 'gender', 'cognition', 'trainee_number', 'completion_method'
    ], errors='ignore')

    # Drop 'score' column from labels
    if 'score' in df_labels.columns:
        df_labels = df_labels.drop(columns=['score'])

    # Preprocess base assessments
    df_assess = preprocess_assessment_data(df_assess, date_column='Source.Name')

    # Convert stem_units to category
    if 'stem_units' in df_assess.columns:
        df_assess['stem_units'] = df_assess['stem_units'].astype('category')

    # Encode conduct_two_assessments
    if 'conduct_two_assessments' in df_assess.columns:
        df_assess['conduct_two_assessments'] = df_assess['conduct_two_assessments'].notna().astype(int)

    # --- Labels for Stage 1 ---
    if stage == 1:
        df_labels = df_labels[df_labels['course_stage'] == 'seamanship_command_substage']
        df_labels = df_labels[df_labels['finished'].isin([0.0, 1.0])]
        df_labels = df_labels[['id', 'finished']]

    # --- Labels for Stage 2 ---
    elif stage == 2:
        df_stage1 = df_labels[df_labels['course_stage'] == 'seamanship_command_substage']
        df_stage1 = df_stage1[df_stage1['finished'].isin([0.0, 1.0])]
        df_stage1 = df_stage1[['id', 'finished']].rename(columns={'finished': 'stage1_finished'})

        df_stage2 = df_labels[df_labels['course_stage'] == 'fundamental_stage']
        df_stage2 = df_stage2[df_stage2['finished'].isin([0.0, 1.0])]
        df_stage2 = df_stage2[['id', 'finished']]

        merged = pd.merge(df_stage1, df_stage2, how='outer', on='id', indicator=True)

        merged['final_stage2_target'] = merged.apply(
            lambda row: 0 if row['stage1_finished'] == 0 else row['finished'], axis=1
        )

        df_labels = merged[['id', 'final_stage2_target']].rename(columns={
            'final_stage2_target': 'finished'
        })
        df_labels = df_labels.dropna(subset=['finished'])
        df_labels['finished'] = df_labels['finished'].astype(int)

    else:
        raise ValueError(f"Invalid stage number {stage}")

    # --- Merge Labels and Assessments ---
    df = pd.merge(df_labels, df_assess, on='id', how='inner')
    df = df.drop(columns=['id'], errors='ignore')

    # Rename finished → target
    df = df.rename(columns={'finished': 'target'})
    df['target'] = df['target'].astype(int)

    return df
