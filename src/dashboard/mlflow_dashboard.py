import streamlit as st
import pandas as pd
from mlflow.tracking import MlflowClient

st.set_page_config(page_title="Navy Completion Dashboard", layout="wide")
st.title("🚢 Navy Course Completion MLflow Dashboard")

# Connect to MLflow
client = MlflowClient()
exp = client.get_experiment_by_name("navy_course_completion")
if exp is None:
    st.error("No experiment named 'navy_course_completion' found.")
    st.stop()

# Fetch runs (sorted by descending F1)
runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    order_by=["metrics.f1_score DESC"],
    max_results=100
)

# Build DataFrame of params & metrics
records = []
for run in runs:
    d = run.data
    records.append({
        "run_id": run.info.run_id,
        "stage": d.params.get("stage", "unknown"),
        "model": d.params.get("model", "ensemble"),
        "f1_score": d.metrics.get("f1_score", None),
        "precision": d.metrics.get("precision", None),
        "recall": d.metrics.get("recall", None),
        "roc_auc": d.metrics.get("roc_auc", None),
        "baseline_f1": d.metrics.get("baseline_f1", None)
    })
df = pd.DataFrame(records)

# Sidebar filters
st.sidebar.header("Filter runs")
st.sidebar.write("Select stage(s) and model(s) to display")
stage_sel = st.sidebar.multiselect("Stage", sorted(df["stage"].unique()), default=sorted(df["stage"].unique()))
model_sel = st.sidebar.multiselect("Model", sorted(df["model"].unique()), default=sorted(df["model"].unique()))

# Filtered DataFrame
df_filtered = df[df["stage"].isin(stage_sel) & df["model"].isin(model_sel)]

# Main view
st.subheader("Runs Table")
st.dataframe(df_filtered.set_index("run_id"), use_container_width=True)

# Key metric plots
col1, col2 = st.columns(2)

with col1:
    st.subheader("F1 Score by Run")
    st.line_chart(df_filtered.set_index("run_id")["f1_score"])

with col2:
    st.subheader("ROC AUC by Run")
    st.line_chart(df_filtered.set_index("run_id")["roc_auc"])

st.markdown("---")
st.subheader("Baseline vs. Model F1")
bar_data = df_filtered[["run_id", "f1_score", "baseline_f1"]].set_index("run_id")
st.bar_chart(bar_data)
