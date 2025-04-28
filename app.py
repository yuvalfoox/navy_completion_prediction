import streamlit as st
import json
import pandas as pd
import plotly.express as px

st.title("🚀 Navy Completion Prediction Dashboard")

# Load model metrics
metrics_file = "output/models_metrics.json"
try:
    with open(metrics_file, "r") as f:
        metrics_all = json.load(f)
except FileNotFoundError:
    st.error("Model metrics not found. Please run training first.")
    st.stop()

model_names = list(metrics_all.keys())
selected_model = st.selectbox("Select model to view", model_names)

# Show metrics
st.subheader("📈 Model Metrics")
metrics = metrics_all[selected_model]
st.metric(label="F1 Score", value=f"{metrics['f1']:.3f}")
st.metric(label="Precision", value=f"{metrics['precision']:.3f}")
st.metric(label="Recall", value=f"{metrics['recall']:.3f}")
st.metric(label="Best Threshold", value=f"{metrics['best_threshold']:.2f}")

# Load feature importances
features_file = f"output/features_importance_{selected_model}.json"
try:
    with open(features_file, "r") as f:
        features = json.load(f)
except FileNotFoundError:
    st.error(f"Feature importances not found for model {selected_model}")
    st.stop()

# Plot feature importances
st.subheader("🔥 Top Feature Importances")
feat_df = pd.DataFrame({
    'feature': list(features.keys()),
    'importance': list(features.values())
})
feat_df = feat_df.sort_values(by="importance", key=abs, ascending=False).head(15)

fig = px.bar(feat_df, x='importance', y='feature', orientation='h', title=f"Top Features for {selected_model}")
st.plotly_chart(fig)
