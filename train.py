
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.metrics import classification_report

def prepare_features(X_train, y_train):
    imp = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns)

    X_train = X_train.loc[:, (X_train != 0).mean() > 0.7]
    var_filter = VarianceThreshold(threshold=0.01)
    X_train = pd.DataFrame(var_filter.fit_transform(X_train), columns=X_train.columns[var_filter.get_support()])
    
    mi = mutual_info_classif(X_train, y_train, random_state=42)
    X_train = X_train.loc[:, mi > 0.001]

    return X_train

def train_models(X_train, y_train):
    lr = LogisticRegression(max_iter=500).fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    ensemble = VotingClassifier(estimators=[('lr', lr), ('rf', rf)], voting='soft').fit(X_train, y_train)
    return lr, rf, ensemble

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
