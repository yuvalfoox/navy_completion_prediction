import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import os

from src.config import *

st.set_page_config(page_title="Navy Completion Prediction", layout="wide")
st.title("🚀 Navy Completion Prediction App")

# --- Sidebar Navigation ---
section = st.sidebar.radio("Navigation", ["Model Analysis", "EDA", "Configuration"])

if section == "EDA":
    st.header("🔍 Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(["📝 Assessments CSV", "📝 Labels CSV", "🔗 Merged Data"])

    with tab1:
        st.subheader("Assessments Data")
        try:
            df_assessments = pd.read_csv("src/data/input/navy_assessments.csv")
        except FileNotFoundError:
            st.error("Assessments data not found.")
            st.stop()

        st.dataframe(df_assessments.head(100))
        st.subheader("Summary Statistics")
        st.dataframe(df_assessments.describe())

        st.subheader("Missing Values (%)")
        missing = df_assessments.isnull().mean() * 100
        missing = missing[missing > 0]
        if not missing.empty:
            st.bar_chart(missing)
        else:
            st.info("No missing values found.")

        st.subheader("Correlations (Top 20)")
        corr = df_assessments.corr(numeric_only=True)
        top_corr = corr.abs().unstack().sort_values(ascending=False).drop_duplicates()
        top_features = list(set([i[0] for i in top_corr.head(20).index] + [i[1] for i in top_corr.head(20).index]))
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr.loc[top_features, top_features], cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.subheader("Distribution Plot")
        numeric_cols = df_assessments.select_dtypes(include=np.number).columns
        selected_col = st.selectbox("Select a numeric column", numeric_cols)
        if selected_col:
            fig = px.histogram(df_assessments, x=selected_col, marginal="box")
            st.plotly_chart(fig)

    with tab2:
        st.subheader("Labels Data")
        try:
            df_labels = pd.read_csv("src/data/input/labels.csv")
        except FileNotFoundError:
            st.error("Labels data not found.")
            st.stop()

        st.dataframe(df_labels.head(100))

        st.subheader("Target Distribution (finished)")
        if "finished" in df_labels.columns:
            fig = px.histogram(df_labels, x="finished", nbins=2, title="Finished Distribution")
            st.plotly_chart(fig)

        st.subheader("Score Distribution (only for analysis)")
        if "score" in df_labels.columns:
            fig = px.histogram(df_labels, x="score", nbins=30, title="Score Distribution")
            st.plotly_chart(fig)

    with tab3:
        st.subheader("Merged Assessments + Labels")
        try:
            df_assessments = pd.read_csv("src/data/input/navy_assessments.csv")
            df_labels = pd.read_csv("src/data/input/labels.csv")
        except FileNotFoundError:
            st.error("Input files not found.")
            st.stop()

        df_merged = pd.merge(df_assessments, df_labels, on="id", how="inner")
        st.dataframe(df_merged.head(100))

        st.subheader("Merged Summary Statistics")
        st.dataframe(df_merged.describe())

        st.subheader("Merged Missing Values (%)")
        missing = df_merged.isnull().mean() * 100
        missing = missing[missing > 0]
        if not missing.empty:
            st.bar_chart(missing)
        else:
            st.info("No missing values found.")

else:
    # === Model Analysis and Configuration stay same ===
    st.header("📊 Model Analysis")

    stage = st.selectbox("Select Stage", ["stage1", "stage2"])
    output_dir = f"output/{stage}"

    # Load model metrics
    metrics_file = os.path.join(output_dir, "models_metrics.json")
    try:
        with open(metrics_file, "r") as f:
            metrics_all = json.load(f)
    except FileNotFoundError:
        st.error("Model metrics not found.")
        st.stop()

    model_names = list(metrics_all.keys())
    selected_model = st.selectbox("Select Model", model_names)

    st.subheader("📈 Model Metrics")
    metrics = metrics_all[selected_model]
    st.metric(label="F1 Score", value=f"{metrics['f1']:.3f}")
    st.metric(label="Precision", value=f"{metrics['precision']:.3f}")
    st.metric(label="Recall", value=f"{metrics['recall']:.3f}")
    st.metric(label="Best Threshold", value=f"{metrics['best_threshold']:.2f}")

    if selected_model != "heuristic":
        try:
            with open(os.path.join(output_dir, f"features_importance_{selected_model}.json"), "r") as f:
                features = json.load(f)
            feat_df = pd.DataFrame({
                'feature': list(features.keys()),
                'importance': list(features.values())
            }).sort_values(by="importance", key=abs, ascending=False).head(15)

            fig_feat = px.bar(feat_df, x='importance', y='feature', orientation='h')
            st.subheader("🔥 Top Feature Importances")
            st.plotly_chart(fig_feat, use_container_width=True)
        except FileNotFoundError:
            st.warning("Feature importances not found.")

    # Confusion Matrix, ROC, PRC, Threshold Tuning (same as before)
