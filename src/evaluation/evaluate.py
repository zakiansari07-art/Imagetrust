import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score, classification_report
from data.dataloader import create_dataloader
from models.detector import create_model

checkpoint_name = "models/checkpoint/resnet_18_ai_detector_best.pth"


def evaluate():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device: ",device)

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    model = create_model()

    checkpoint = torch.load(checkpoint_name, map_location=device)



    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_generators = []

    _, validation_dataloader = create_dataloader()
    with torch.no_grad():
        for images, labels in validation_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            logits = logits.squeeze(1)

            labels = labels.cpu().numpy()

            all_labels.extend(labels)

            probabilities = torch.sigmoid(logits)

            all_probabilities.extend(probabilities.cpu().numpy())

            prediction = (probabilities >= 0.5).float()

            all_predictions.extend(prediction.cpu().numpy())

    # Metrics

    accuracy = accuracy_score(all_labels, all_predictions)

    recall = recall_score(all_labels, all_predictions)

    f1 = f1_score(all_labels, all_predictions)

    cn_matrix = confusion_matrix(all_labels, all_predictions)

    roc_auc = roc_auc_score(all_labels, all_probabilities)

    report = classification_report(all_labels, all_predictions,target_names=["Real", "Fake"])
    precision = precision_score(all_labels, all_predictions)

    print(f"""Accuracy Score:" {accuracy}
        "Precision Score:" {precision}
        "F1 Score:" {f1}
        "roc_auc_score:" {roc_auc}
        "recall:" {recall}
        "confusion_matrix:" {cn_matrix}""")

    print(report)


if __name__ == "__main__":
    evaluate()

