import logging
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

def plot_confusion_matrices(models: dict, X_test, y_test):
    logger.info("Plotting confusion matrices")
    plt.figure(figsize=(5 * len(models), 4))
    for i, (name, model) in enumerate(models.items(), 1):
        cm = confusion_matrix(y_test, model.predict(X_test))
        plt.subplot(1, len(models), i)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(name)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

def plot_roc_curves(models: dict, X_test, y_test):
    logger.info("Plotting ROC curves")
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")
    plt.plot([0,1], [0,1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

def plot_shap_summary(model, X_test):
    logger.info("Generating SHAP summary (bar) plot")
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test)
        vals = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        shap.summary_plot(vals, X_test, plot_type="bar")
    except Exception as e:
        logger.error("SHAP TreeExplainer failed, fallback to KernelExplainer: %s", e)
        explainer = shap.KernelExplainer(model.predict_proba, X_test.iloc[:100])
        shap_vals = explainer.shap_values(X_test.iloc[:100])
        shap.summary_plot(shap_vals[1], X_test.iloc[:100], plot_type="bar")
