#!/usr/bin/env bash
set -e

echo "🚀 Installing requirements..."
pip install -r requirements.txt

echo "🚀 Setting PYTHONPATH..."
export PYTHONPATH=.

echo "🚀 Running full Navy Completion Prediction pipeline..."
python -m src.running.run_pipeline \
  --data-file src/data/input/navy_assessments.csv \
  --labels-file src/data/input/labels.csv

echo "🚀 Launching Streamlit dashboard..."
streamlit run app.py
