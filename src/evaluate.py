from sklearn.metrics import(
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate the model using various metrics
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics={
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "f1_score": round(f1_score(y_test, y_pred, average='binary'), 4),
    }

    return metrics, y_pred

def print_evaluation(metrics: dict, y_test, y_pred):
    """
    Print the evaluation results in a readable format
    """
    
    print("\n" + "="*40)
    print("MODEL EVALUATION")
    for metric, value in metrics.items():
        print(f"  {metric:<12}: {value}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("="*40 + "\n")

