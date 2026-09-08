import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from data.dataloader_defactify import create_generator_task_eval_dataloader
from models.detector import create_defactify_generator_model
import numpy as np
from datasets import load_dataset
from collections import Counter
from pathlib import Path
from datetime import datetime
import json


project_root = Path.cwd()


path_name = project_root.parent

model_checkpoint = "multi_task_defactify_last_checkpoint.pth"


file_name = "eval_multitask_last_ch.json"


def evaluate_generator_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))


    model = create_defactify_generator_model()

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

   
    
    all_predictions = []
    all_probabilities = []
    all_generators = []

    validation_dataloader, _ = create_generator_task_eval_dataloader()

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    with torch.no_grad():

        for images, generators in validation_dataloader:

            images = images.to(device=device, non_blocking=True)
                         
            
            generators = generators.numpy()
           

            logits = model(images)

            

            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            

            
            

            all_probabilities.extend(probabilities.cpu().numpy())
            

            all_predictions.extend(predictions.cpu().numpy())
            

            all_generators.extend(generators)

   
    
    print("Predictions:", len(all_predictions))
    print("Probabilities:", len(all_probabilities))
    print("Generators:", len(all_generators))

   
    # ---------------------------------------------------------
    # Convert to numpy arrays once
    # ---------------------------------------------------------
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_generators = np.array(all_generators)


    timestamp = datetime.now()
    
        # ---------------------------------------------------------
        # Create evaluation result directory
        # ---------------------------------------------------------
    results_dir = Path("src/evaluation/evaluation_results")
    
    dir_name = (
            f"{checkpoint_name.stem}_"
            f"{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}"
        )
    
    file_dir = results_dir / dir_name
    file_dir.mkdir(parents=True, exist_ok=True)

    generator_names = ["SD21", "SDXL", "SD3", "DALLE3", "Midjourney"]
    results = []

    for generator_id, generator_name in enumerate(
        generator_names
    ):


        binary_predictions = (all_predictions == generator_id).astype(int)
        binary_generators = (all_generators == generator_id).astype(int)

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
            labels=[f"{generator_name}", "other_generators"]
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