import streamlit as st
import pandas as pd
from mlflow.tracking import MlflowClient

st.set_page_config(page_title="Navy Completion Dashboard", layout="wide")
st.title("🚢 Navy Course Completion MLflow Dashboard")

client = MlflowClient()
exp = client.get_experiment_by_name("navy_course_completion")
if exp is None:
    st.error("Experiment 'navy_course_completion' not found.")
    st.stop()

runs = client.search_runs(exp.experiment_id, order_by=["metrics.f1_score DESC"], max_results=100)
records = []
for run in runs:
    d = run.data
    records.append({
        "run_id": run.info.run_id,
        "stage": d.params.get("stage",""),
        "model": d.params.get("model","ensemble"),
        "f1_score": d.metrics.get("f1_score", 0),
        "precision": d.metrics.get("precision",0),
        "recall": d.metrics.get("recall",0),
        "roc_auc": d.metrics.get("roc_auc",0),
        "baseline_f1": d.metrics.get("baseline_f1",0)
    })

df = pd.DataFrame(records)
stage_sel = st.sidebar.multiselect("Stage", sorted(df.stage.unique()), default=df.stage.unique())
model_sel = st.sidebar.multiselect("Model", sorted(df.model.unique()), default=df.model.unique())

df_f = df[df.stage.isin(stage_sel) & df.model.isin(model_sel)]
st.subheader("Runs")
st.dataframe(df_f.set_index("run_id"), use_container_width=True)

st.subheader("F1 over Runs")
st.line_chart(df_f.set_index("run_id")["f1_score"])

st.subheader("ROC AUC over Runs")
st.line_chart(df_f.set_index("run_id")["roc_auc"])

st.subheader("Baseline vs Model F1")
st.bar_chart(df_f.set_index("run_id")[["f1_score","baseline_f1"]])
