#!/usr/bin/env bash
# run.sh – install, launch dashboards, run pipeline, and build Evidently report
# Usage: ./run.sh [data_file] [labels_file] ["stages"]
# Defaults:
#   data_file: src/data/input/navy_assessments.csv
#   labels_file: src/data/input/labels.csv
#   stages: "1 2"

DATA_FILE="${1:-src/data/input/navy_assessments.csv}"
LABELS_FILE="${2:-src/data/input/labels.csv}"
STAGES="${3:-1 2}"

echo "Installing dependencies..."
pip install -r requirements.txt

echo
echo "🚀 Launching MLflow UI at http://localhost:5000"
mlflow ui --port 5000 --host 0.0.0.0 &

echo
for STAGE in $STAGES; do
  echo "=== Running Stage $STAGE ==="
  python -m src.running.run_pipeline \
    --data-file "$DATA_FILE" \
    --labels-file "$LABELS_FILE" \
    --stage "$STAGE"
  echo
done

echo "🔍 Generating Evidently report (ClassificationPerformanceTab)…"
python - << 'EOF'
import pandas as pd
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import ClassificationPerformanceTab

# Adjust paths if needed:
#   - test set with true labels
#   - predictions file with a 'predicted' column
test = pd.read_csv("src/data/input/navy_assessments.csv")  # replace with your actual test-split CSV
preds = pd.read_csv("predictions_stageX.csv")            # generate this in your pipeline

# Create and save the report
dashboard = Dashboard(tabs=[ClassificationPerformanceTab()])
dashboard.calculate(test, preds['predicted'], column_mapping={
    'target': 'target',          # true label column
    'prediction': 'predicted'    # predicted label column
})
dashboard.save("evidently_report.html")
EOF

echo
echo "✅ Done!"
echo "- MLflow UI: http://localhost:5000"  
echo "- Evidently report: evidently_report.html"
