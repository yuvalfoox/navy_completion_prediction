# Navy Course Completion Prediction

A modular machine-learning pipeline to predict trainee course completion for naval training stages, with end-to-end support from data loading through interpretability.

## 📋 Overview

- **Goal**: Predict whether a trainee will complete Stage 1 or Stage 2 of the naval course.
- **Pipeline**:
  1. **Data loading** (`src/data/load_data.py`): read raw Excel, extract assessment counter, label targets per stage.
  2. **Feature engineering** (`src/features/engineer.py`): create time-aware and performance-normalized features.
  3. **Feature selection & preprocessing** (`src/models/train.py`): impute, filter, and select predictive features.
  4. **Model training** (`src/models/train.py`): train Logistic Regression, Random Forest, and an ensemble.
  5. **Evaluation & visualization** (`src/visualization/plots.py`): confusion matrices, ROC/PR curves, SHAP interpretability.
  6. **Notebook orchestration** (`notebooks/main_stage1_pipeline.ipynb`): run full Stage 1 flow; easily port to Stage 2.

# Navy Course Completion Prediction

## Setup & Run

```bash
git clone <repo>
cd navy_completion_prediction
chmod +x run.sh
./run.sh

