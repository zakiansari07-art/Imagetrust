import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from data.dataloader import create_generator_dataloader
from models.detector import create_generator_model
import numpy as np
from datasets import load_dataset
from collections import Counter
from pathlib import Path
from datetime import datetime
import json


project_root = Path.cwd()

path_name = project_root.parent

model_checkpoint = "gen_model_lr_rate_change_best_validation.pth"


file_name = "gen_model_eval.json"


def evaluate_generator_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))


    model = create_generator_model()

    checkpoint_name = Path(f"models/checkpoint/{model_checkpoint}")
    

    checkpoint = torch.load(
        checkpoint_name,
        map_location=device
    )

   
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device=device, non_blocking=True)
    model.eval()

   
    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_generators = []

    _, validation_dataloader = create_generator_dataloader()

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    with torch.no_grad():

        for images, labels, generators in validation_dataloader:

            images = images.to(device=device, non_blocking=True)
                         
            labels = labels.numpy()
            generators = generators.numpy()
           

            logits = model(images)

            

            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            

            
            all_labels.extend(labels)

            all_probabilities.extend(probabilities.cpu().numpy())
            

            all_predictions.extend(predictions.cpu().numpy())
            

            all_generators.extend(generators)

   
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

    generator_names = (
        hf_dataset["validation"]
        .features["generator"].names
    )

    print("\nGenerator mapping:")
    print(generator_names)

    generator_counts = Counter(all_generators)

    for generator_id, name in enumerate(
        generator_names
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


    timestamp = datetime.now()
    
        # ---------------------------------------------------------
        # Create evaluation result directory
        # ---------------------------------------------------------
    results_dir = Path("src/evaluation/evaluation_results_generator_model")
    
    dir_name = (
            f"{checkpoint_name.stem}_"
            f"{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}"
        )
    
    file_dir = results_dir / dir_name
    file_dir.mkdir(parents=True, exist_ok=True)

    
    results = []

    for generator_id, generator_name in enumerate(
        generator_names
    ):

        # Skip the real class: 0
        if generator_id == 0:
            continue
        # Skip classes not in the dataset (SD14)
        if not np.any(generator_id == all_generators):
            continue

     

        #Select real and generaotor images
        mask = (all_generators == 0) | (all_generators == generator_id)
        

        labels = all_labels[mask]
        predictions = all_predictions[mask] 
        
        generators = all_generators[mask]

        binary_predictions = (predictions == generator_id).astype(int)
        binary_generators = (generators == generator_id).astype(int)

        binary_probabilities = all_probabilities[mask, generator_id]

      
        accuracy = accuracy_score(
            binary_generators,
            binary_predictions
        )

        precision = precision_score(
            binary_generators,
            binary_predictions,
            zero_division=0
        )

        recall = recall_score(
            binary_generators,
            binary_predictions,
            zero_division=0
        )

        f1 = f1_score(
            binary_generators,
            binary_predictions,
            zero_division=0
        )

        # ROC-AUC requires both classes to exist
        if len(np.unique(binary_generators)) > 1:

            roc_auc = roc_auc_score(
                binary_generators,
                binary_probabilities
            )

        else:

            roc_auc = None

        # -----------------------------------------------------
        # Confusion matrix
        # -----------------------------------------------------
        cm = confusion_matrix(
            binary_generators,
            binary_predictions,
            labels=[0, 1]
        )

        # -----------------------------------------------------
        # Store result
        # -----------------------------------------------------
        results.append(
            {
                
                "Generator": generator_name,
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
    with open(file_dir/file_name, "w") as f:

        json.dump(
            results,
            f,
            indent=4
        )

  
    print("Evaluataion Complete!")
  


if __name__ == "__main__":
    evaluate_generator_model()