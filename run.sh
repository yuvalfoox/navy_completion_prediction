#!/usr/bin/env bash
set -e

# Install requirements
echo "🚀 Installing requirements..."
pip install -r requirements.txt

# Export PYTHONPATH so src/ is visible
export PYTHONPATH=.

# Create output folder if not exists
mkdir -p output

# Run pipeline
echo "🚀 Running Navy Completion Prediction pipeline..."
python -m src.running.run_pipeline \
  --data-file src/data/input/navy_assessments.csv \
  --labels-file src/data/input/labels.csv

# Launch Streamlit Dashboard
echo "🚀 Launching Streamlit Dashboard..."
streamlit run app.py
