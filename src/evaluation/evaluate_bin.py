import json
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    classification_report,
)
from data.dataloader_defactify import create_binary_task_eval_dataloader
from models.detector import create_model


CHECKPOINT_NAME = "models/checkpoint/bin_task_defactify_best_validation.pth"

RESULTS_DIR = Path(
    "src/evaluation/evaluation_results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

HISTORY_FILE = RESULTS_DIR / "evaluation_history_defactify.csv"


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

    # ---------------------------------------------------------
    # 3. Create validation dataloader
    # ---------------------------------------------------------

    validation_dataloader, test_dataloader = create_binary_task_eval_dataloader()

    # ---------------------------------------------------------
    # 4. Store predictions
    # ---------------------------------------------------------

    all_labels = []
    all_predictions = []
    all_probabilities = []

    threshold = 0.5

    # ---------------------------------------------------------
    # 5. Run evaluation
    # ---------------------------------------------------------

    with torch.no_grad():

        for images, labels in test_dataloader:

            images = images.to(device)

            # Labels don't need to be on GPU
            labels = labels.numpy()

            # Model prediction
            logits = model(images)

            # Convert [batch_size, 1]
            # to [batch_size]
            logits = logits.squeeze(1)

            # Convert logits to probabilities
            probabilities = torch.sigmoid(logits)

            # Convert probabilities to predictions
            predictions = (
                probabilities >= threshold
            ).float()

            # Store results
            all_labels.extend(labels)

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    # ---------------------------------------------------------
    # 6. Convert results to NumPy arrays
    # ---------------------------------------------------------

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

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
        target_names=["Real", "Fake"],
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

    run_dir = RESULTS_DIR / run_name

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # 12. Create metrics dictionary
    # ---------------------------------------------------------

    metrics = {
        "timestamp": timestamp.isoformat(
            timespec="seconds"
        ),
        "checkpoint": CHECKPOINT_NAME,
        "device": str(device),
        "threshold": threshold,
        "num_samples": len(all_labels),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": (
            float(roc_auc)
            if roc_auc is not None
            else None
        ),
        "confusion_matrix": cn_matrix.tolist()
    }

    # ---------------------------------------------------------
    # 13. Save metrics as JSON
    # ---------------------------------------------------------

    metrics_file = run_dir / "metrics.json"

    with open(
        metrics_file,
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    # ---------------------------------------------------------
    # 14. Save confusion matrix
    # ---------------------------------------------------------

    confusion_matrix_file = (
        run_dir / "confusion_matrix.json"
    )

    with open(
        confusion_matrix_file,
        "w"
    ) as f:

        json.dump(
            cn_matrix.tolist(),
            f,
            indent=4
        )

    # ---------------------------------------------------------
    # 15. Save classification report
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 16. Append results to evaluation history
    # ---------------------------------------------------------

    history_exists = HISTORY_FILE.exists()

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

        # Only write header for a new CSV
        if not history_exists:
            writer.writeheader()

        # Only put history fields into CSV
        history_metrics = {
            field: metrics[field]
            for field in history_fields
        }

        writer.writerow(history_metrics)

    # ---------------------------------------------------------
    # 17. Print results
    # ---------------------------------------------------------

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

    print("\nConfusion Matrix:")
    print(cn_matrix)

    print("\nClassification Report:")

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=["Real", "Fake"],
            zero_division=0
        )
    )

    # ---------------------------------------------------------
    # 18. Print saved locations
    # ---------------------------------------------------------

    print(
        f"\nResults saved to:\n{run_dir}"
    )

    print(
        f"History saved to:\n{HISTORY_FILE}"
    )


if __name__ == "__main__":
    evaluate()