
import json
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
    ConfusionMatrixDisplay,
)

from data.dataloader_defactify import create_binary_task_eval_dataloader
from models.detector import create_model


# =========================================================
# Configuration
# =========================================================

CHECKPOINT_NAME = (
    "models/checkpoint/bin_task_defactify_best_validation.pth"
)

RESULTS_DIR = Path(
    "src/evaluation/evaluation_results"
)

HISTORY_FILE = (
    RESULTS_DIR / "evaluation_history_defactify.csv"
)

THRESHOLD = 0.5


# =========================================================
# Create results directory
# =========================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# Evaluation
# =========================================================

def evaluate():

    # ---------------------------------------------------------
    # 1. Setup device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ---------------------------------------------------------
    # 2. Load model
    # ---------------------------------------------------------

    print("\nLoading model...")

    model = create_model()

    checkpoint = torch.load(
        CHECKPOINT_NAME,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print("Checkpoint loaded:", CHECKPOINT_NAME)

    # ---------------------------------------------------------
    # 3. Create validation/test dataloaders
    # ---------------------------------------------------------

    validation_dataloader, test_dataloader = (
        create_binary_task_eval_dataloader()
    )

    # ---------------------------------------------------------
    # 4. Store predictions
    # ---------------------------------------------------------

    all_labels = []
    all_predictions = []
    all_probabilities = []

    # ---------------------------------------------------------
    # 5. Run evaluation
    # ---------------------------------------------------------

    print("\nRunning evaluation...")

    with torch.no_grad():

        for images, labels in test_dataloader:

            images = images.to(device)

            # Labels don't need to be on GPU
            labels = labels.numpy()

            # Model output
            logits = model(images)

            # Convert [batch_size, 1]
            # to [batch_size]
            logits = logits.squeeze(1)

            # Convert logits to probabilities
            probabilities = torch.sigmoid(logits)

            # Convert probabilities to binary predictions
            predictions = (
                probabilities >= THRESHOLD
            ).float()

            # Store results
            all_labels.extend(
                labels
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    # ---------------------------------------------------------
    # 6. Convert results to NumPy arrays
    # ---------------------------------------------------------

    all_labels = np.array(
        all_labels
    ).astype(int)

    all_predictions = np.array(
        all_predictions
    ).astype(int)

    all_probabilities = np.array(
        all_probabilities
    )

    print(
        "\nNumber of samples:",
        len(all_labels)
    )

    # ---------------------------------------------------------
    # 7. Check class distribution
    # ---------------------------------------------------------

    unique_classes, class_counts = np.unique(
        all_labels,
        return_counts=True
    )

    print("\nClass distribution:")

    for class_id, count in zip(
        unique_classes,
        class_counts
    ):

        class_name = (
            "Real"
            if class_id == 0
            else "Fake"
        )

        print(
            f"{class_name} ({class_id}): {count}"
        )

    # ---------------------------------------------------------
    # 8. Calculate metrics
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    # ROC-AUC requires both classes
    if len(np.unique(all_labels)) == 2:

        roc_auc = roc_auc_score(
            all_labels,
            all_probabilities
        )

    else:

        roc_auc = None

    # ---------------------------------------------------------
    # 9. Confusion matrix
    # ---------------------------------------------------------

    cn_matrix = confusion_matrix(
        all_labels,
        all_predictions
    )

    # ---------------------------------------------------------
    # 10. Classification report
    # ---------------------------------------------------------

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=[
            "Real",
            "Fake"
        ],
        output_dict=True,
        zero_division=0
    )

    # ---------------------------------------------------------
    # 11. Create evaluation metadata
    # ---------------------------------------------------------

    timestamp = datetime.now()

    checkpoint_path = Path(
        CHECKPOINT_NAME
    )

    run_name = (
        f"{checkpoint_path.stem}_"
        f"{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    run_dir = (
        RESULTS_DIR / run_name
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\nSaving results to:",
        run_dir
    )

    # =========================================================
    # 12. Save ROC curve
    # =========================================================

    roc_curve_file = None

    if len(np.unique(all_labels)) == 2:

        fpr, tpr, thresholds = roc_curve(
            all_labels,
            all_probabilities
        )

        roc_curve_file = (
            run_dir / "roc_curve.png"
        )

        plt.figure(
            figsize=(8, 6)
        )

        plt.plot(
            fpr,
            tpr,
            label=f"ROC curve (AUC = {roc_auc:.4f})"
        )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Random classifier"
        )

        plt.xlabel(
            "False Positive Rate"
        )

        plt.ylabel(
            "True Positive Rate"
        )

        plt.title(
            "ROC Curve - Real vs AI Generated"
        )

        plt.legend(
            loc="lower right"
        )

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            roc_curve_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "ROC curve saved:",
            roc_curve_file
        )

    # =========================================================
    # 13. Save confusion matrix image
    # =========================================================

    confusion_matrix_file = (
        run_dir / "confusion_matrix.png"
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cn_matrix,
        display_labels=[
            "Real",
            "Fake"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    display.plot(
        ax=ax,
        values_format="d"
    )

    ax.set_title(
        "Confusion Matrix - Real vs AI Generated"
    )

    plt.tight_layout()

    plt.savefig(
        confusion_matrix_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Confusion matrix saved:",
        confusion_matrix_file
    )

    # =========================================================
    # 14. Save predictions CSV
    # =========================================================

    predictions_file = (
        run_dir / "predictions.csv"
    )

    with open(
        predictions_file,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "true_label",
            "true_class",
            "predicted_label",
            "predicted_class",
            "fake_probability"
        ])

        for true_label, prediction, probability in zip(
            all_labels,
            all_predictions,
            all_probabilities
        ):

            true_class = (
                "Real"
                if true_label == 0
                else "Fake"
            )

            predicted_class = (
                "Real"
                if prediction == 0
                else "Fake"
            )

            writer.writerow([
                int(true_label),
                true_class,
                int(prediction),
                predicted_class,
                float(probability)
            ])

    print(
        "Predictions saved:",
        predictions_file
    )

    # =========================================================
    # 15. Create metrics dictionary
    # =========================================================

    metrics = {

        "timestamp": timestamp.isoformat(
            timespec="seconds"
        ),

        "checkpoint": CHECKPOINT_NAME,

        "device": str(device),

        "threshold": THRESHOLD,

        "num_samples": len(all_labels),

        "accuracy": float(
            accuracy
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1": float(
            f1
        ),

        "roc_auc": (
            float(roc_auc)
            if roc_auc is not None
            else None
        ),

        "confusion_matrix": (
            cn_matrix.tolist()
        )
    }

    # =========================================================
    # 16. Save metrics JSON
    # =========================================================

    metrics_file = (
        run_dir / "metrics.json"
    )

    with open(
        metrics_file,
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    print(
        "Metrics saved:",
        metrics_file
    )

    # =========================================================
    # 17. Save confusion matrix JSON
    # =========================================================

    confusion_matrix_json_file = (
        run_dir / "confusion_matrix.json"
    )

    with open(
        confusion_matrix_json_file,
        "w"
    ) as f:

        json.dump(
            cn_matrix.tolist(),
            f,
            indent=4
        )

    print(
        "Confusion matrix JSON saved:",
        confusion_matrix_json_file
    )

    # =========================================================
    # 18. Save classification report
    # =========================================================

    report_file = (
        run_dir / "classification_report.json"
    )

    with open(
        report_file,
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    print(
        "Classification report saved:",
        report_file
    )

    # =========================================================
    # 19. Append results to evaluation history
    # =========================================================

    history_exists = (
        HISTORY_FILE.exists()
    )

    history_fields = [
        "timestamp",
        "checkpoint",
        "device",
        "threshold",
        "num_samples",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    with open(
        HISTORY_FILE,
        "a",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=history_fields
        )

        if not history_exists:
            writer.writeheader()

        history_metrics = {
            field: metrics[field]
            for field in history_fields
        }

        writer.writerow(
            history_metrics
        )

    # =========================================================
    # 20. Print results
    # =========================================================

    print("\nEvaluation Results")
    print("------------------")

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    if roc_auc is not None:

        print(
            f"ROC-AUC  : {roc_auc:.4f}"
        )

    else:

        print(
            "ROC-AUC  : N/A "
            "(only one class present)"
        )

    # =========================================================
    # 21. Print confusion matrix
    # =========================================================

    print("\nConfusion Matrix:")

    print(cn_matrix)

    # =========================================================
    # 22. Print classification report
    # =========================================================

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=[
                "Real",
                "Fake"
            ],
            zero_division=0
        )
    )

    # =========================================================
    # 23. Print saved locations
    # =========================================================

    print(
        "\nResults saved to:"
    )

    print(run_dir)

    print(
        "\nHistory saved to:"
    )

    print(HISTORY_FILE)

    print("\nGenerated files:")

    for file in sorted(run_dir.iterdir()):

        print(
            f"  - {file.name}"
        )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    evaluate()
