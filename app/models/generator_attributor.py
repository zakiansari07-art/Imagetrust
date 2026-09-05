from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from models.detector import create_generator_model


GENERATOR_LABELS = [
    "real",
    "ADM",
    "BigGAN",
    "GLIDE",
    "Midjourney",
    "SD14",
    "SD15",
    "VQDM",
    "Wukong",
]


class GeneratorAttributor:
    """Predicts the source most similar to the model's known generators."""

    def __init__(self, checkpoint_path: str):

        path = Path(checkpoint_path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = create_generator_model()

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def predict(self, image_path: str) -> dict:
      
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)

        probabilities_by_label = {
            label: round(probability.item(), 4)
            for label, probability in zip(GENERATOR_LABELS, probabilities)
        }

        best_index = torch.argmax(probabilities).item()
        best_label = GENERATOR_LABELS[best_index]
        confidence = probabilities[best_index].item()

        return {
            "label": best_label,
            "confidence": round(confidence, 4),
            "probabilities": probabilities_by_label,
            "model_name": "resnet18_generator_attributor",
        }