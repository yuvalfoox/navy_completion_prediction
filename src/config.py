# src/config.py

# Feature Engineering Parameters
missing_threshold = 30.0   # % missing allowed
zero_threshold = 35.0      # % zeros allowed
var_threshold = 0.005      # Variance threshold
mi_threshold = 0.0001      # Mutual Information minimum

# Correlation Filtering
corr_threshold = 0.9

# Data Split Ratios
train_frac = 0.7
val_frac = 0.15
test_frac = 0.15

# Model Training
cv_folds = 6
random_search_iter = 100
random_state = 42
