from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(
    y_test,
    y_pred
):
    """
    Calculate standard classification metrics.
    """

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    confusion = confusion_matrix(
        y_test,
        y_pred
    )

    report = classification_report(
        y_test,
        y_pred
    )

    return accuracy, confusion, report