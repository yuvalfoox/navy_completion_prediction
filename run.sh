#!/usr/bin/env bash
set -e

# Install requirements
pip install -r requirements.txt

# Export path
export PYTHONPATH=.

# Run pipeline
python -m src.running.run_pipeline \
  --data-file src/data/input/navy_assessments.csv \
  --labels-file src/data/input/labels.csv
