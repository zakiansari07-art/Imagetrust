import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from data.dataloader import create_generator_evaluation_dataloader
from models.detector import create_model
import numpy as np
from datasets import load_dataset
from collections import Counter
from pathlib import Path
from datetime import datetime
import json
import csv


def evaluate_gen():

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    # ---------------------------------------------------------
    # Load model and checkpoint
    # ---------------------------------------------------------
    model = create_model()

    checkpoint_name = "models/checkpoint/last_checkpoint.pth"
    checkpoint_path = Path(checkpoint_name)

    checkpoint = torch.load(
        checkpoint_name,
        map_location=device
    )

    timestamp = datetime.now()

    # ---------------------------------------------------------
    # Create evaluation result directory
    # ---------------------------------------------------------
    results_dir = Path("src/evaluation/evaluation_results_generator")

    dir_name = (
        f"{checkpoint_path.stem}_"
        f"{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    file_dir = results_dir / dir_name
    file_dir.mkdir(parents=True, exist_ok=True)

    file_name = file_dir / "generator_evaluation.json"

    history_filename = results_dir / "evaluation_generator_history.csv"

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    # ---------------------------------------------------------
    # Store predictions
    # ---------------------------------------------------------
    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_generators = []

    validation_dataloader = create_generator_evaluation_dataloader()

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    with torch.no_grad():

        for images, labels, generators in validation_dataloader:

            images = images.to(device)

            # Labels and generators don't need to be on GPU
            labels = labels.numpy()
            generators = generators.numpy()

            # Model inference
            logits = model(images)
            logits = logits.squeeze(1)

            # Convert logits -> probability
            probabilities = torch.sigmoid(logits)

            # Probability -> binary prediction
            predictions = (
                probabilities >= 0.5
            ).float()

            # Store results
            all_labels.extend(labels)

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_generators.extend(generators)

    # ---------------------------------------------------------
    # Sanity checks
    # ---------------------------------------------------------
    print("Labels:", len(all_labels))
    print("Predictions:", len(all_predictions))
    print("Probabilities:", len(all_probabilities))
    print("Generators:", len(all_generators))

    # ---------------------------------------------------------
    # Generator information from Hugging Face dataset
    # ---------------------------------------------------------
    hf_dataset = load_dataset(
        "TheKernel01/Tiny-GenImage"
    )

    generator_feature = (
        hf_dataset["validation"]
        .features["generator"]
    )

    print("\nGenerator mapping:")
    print(generator_feature.names)

    generator_counts = Counter(all_generators)

    for generator_id, name in enumerate(
        generator_feature.names
    ):
        print(
            f"ID {generator_id}: "
            f"{name} | "
            f"Count: {generator_counts.get(generator_id, 0)}"
        )

    # ---------------------------------------------------------
    # Convert to numpy arrays once
    # ---------------------------------------------------------
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_generators = np.array(all_generators)

    # ---------------------------------------------------------
    # Generators to evaluate
    #
    # Assumption:
    # generator ID 0 = real images
    # generator IDs > 0 = AI generators
    #
    # Change this if your dataset mapping is different.
    # ---------------------------------------------------------
    results = []

    for generator_id, generator_name in enumerate(
        generator_feature.names
    ):

        # Skip real class
        if generator_id == 0:
            continue

        # Skip SD14 if its dataset ID is 5
        if generator_id == 5:
            continue

        # -----------------------------------------------------
        # Select:
        #   real images (generator 0)
        #   current AI generator
        # -----------------------------------------------------
        mask = (
            (all_generators == 0)
            | (all_generators == generator_id)
        )

        labels = all_labels[mask]
        predictions = all_predictions[mask]
        probabilities = all_probabilities[mask]

        # -----------------------------------------------------
        # Metrics
        # -----------------------------------------------------
        accuracy = accuracy_score(
            labels,
            predictions
        )

        precision = precision_score(
            labels,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            labels,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            labels,
            predictions,
            zero_division=0
        )

        # ROC-AUC requires both classes to exist
        if len(np.unique(labels)) == 2:

            roc_auc = roc_auc_score(
                labels,
                probabilities
            )

        else:

            roc_auc = None

        # -----------------------------------------------------
        # Confusion matrix
        # -----------------------------------------------------
        cm = confusion_matrix(
            labels,
            predictions
        )

        # -----------------------------------------------------
        # Store result
        # -----------------------------------------------------
        results.append(
            {
                "timestamp": timestamp.isoformat(
                    timespec="seconds"
                ),
                "checkpoint": checkpoint_path.name,
                "Generator": generator_name,
                "Generator ID": int(generator_id),
                "Accuracy": float(accuracy),
                "Precision": float(precision),
                "Recall": float(recall),
                "F1": float(f1),
                "ROC-AUC": (
                    float(roc_auc)
                    if roc_auc is not None
                    else None
                ),
                "Confusion Matrix": cm.tolist(),
            }
        )

    # ---------------------------------------------------------
    # Save detailed JSON
    # ---------------------------------------------------------
    with open(file_name, "w") as f:

        json.dump(
            results,
            f,
            indent=4
        )

    # ---------------------------------------------------------
    # Save historical CSV
    # ---------------------------------------------------------
    history_fields = [
        "timestamp",
        "checkpoint",
        "Generator",
        "Generator ID",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
    ]

    # Check whether the file existed BEFORE opening it
    file_exists = history_filename.exists()

    with open(
        history_filename,
        "a",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=history_fields
        )

        # Write header only for a new CSV
        if not file_exists:
            writer.writeheader()

        # Write one row per generator
        for result in results:

            history_result = {
                field: result[field]
                for field in history_fields
            }

            writer.writerow(history_result)

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------
    for result in results:

        print("\n" + "=" * 50)

        print(result["Generator"])

        print("=" * 50)

        print(
            f"Accuracy:  {result['Accuracy']:.4f}"
        )

        print(
            f"Precision: {result['Precision']:.4f}"
        )

        print(
            f"Recall:    {result['Recall']:.4f}"
        )

        print(
            f"F1:        {result['F1']:.4f}"
        )

        if result["ROC-AUC"] is not None:

            print(
                f"ROC-AUC:   {result['ROC-AUC']:.4f}"
            )

        else:

            print("ROC-AUC:   N/A")

        print("Confusion Matrix:")

        print(
            np.array(
                result["Confusion Matrix"]
            )
        )

    print(
        f"\nDetailed results saved to:\n{file_name}"
    )

    print(
        f"History saved to:\n{history_filename}"
    )


if __name__ == "__main__":
    evaluate_gen()