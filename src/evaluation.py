import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def evaluate_thresholds(y_test, y_probability):
    """
    Evaluate model performance across different
    probability decision thresholds.
    """

    threshold_results = []

    thresholds = np.arange(
        0.20,
        0.51,
        0.01
    )

    for threshold in thresholds:

        y_threshold_pred = (
            y_probability >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            y_threshold_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_threshold_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_threshold_pred,
            zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_threshold_pred
        ).ravel()

        threshold_results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positives": fp,
            "false_negatives": fn
        })

    return pd.DataFrame(threshold_results)

def save_threshold_report(
    threshold_df,
    report_path,
    plot_path
):
    """
    Save threshold results and create
    a precision/recall/F1 trade-off plot.
    """

    best_threshold = threshold_df.loc[
        threshold_df["f1"].idxmax()
    ]

    threshold_df.to_csv(
        report_path,
        index=False
    )

    selected_threshold = best_threshold[
        "threshold"
    ]

    plt.figure(figsize=(10, 6))

    plt.plot(
        threshold_df["threshold"],
        threshold_df["precision"],
        label="Precision"
    )

    plt.plot(
        threshold_df["threshold"],
        threshold_df["recall"],
        label="Recall"
    )

    plt.plot(
        threshold_df["threshold"],
        threshold_df["f1"],
        label="F1"
    )

    plt.axvline(
        selected_threshold,
        linestyle="--",
        label=(
            f"Selected threshold "
            f"({selected_threshold:.2f})"
        )
    )

    plt.xlabel("Decision threshold")
    plt.ylabel("Score")

    plt.title(
        "Precision, Recall and F1 Across Decision Thresholds"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        plot_path,
        dpi=300
    )

    plt.close()

    return best_threshold