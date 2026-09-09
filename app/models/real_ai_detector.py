from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from models.detector import create_model


class RealAIDetector:
    """Loads the trained real-vs-AI model and predicts one image at a time."""

    def __init__(self, checkpoint_path: str):

        path = Path(checkpoint_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Rebuild the same ResNet-18 architecture used during training.
        self.model = create_model()

        # Load the learned weights from your trusted checkpoint.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.to(self.device)
        self.model.eval()  # inference mode: no training behaviour

        # This must match your training preprocessing exactly.
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def predict(self, image_path: str ) -> dict:
        """Return a clear, app-friendly prediction for one image."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.model(image_tensor).squeeze()
            ai_probability = torch.sigmoid(logit).item()

        real_probability = 1 - ai_probability

        
        confidence = max(ai_probability, real_probability)

        return {
            "label": None,
            "confidence": round(confidence, 4),
            "probabilities": {
                "real": round(real_probability, 4),
                "ai_generated": round(ai_probability, 4),
            },
            "model_name": "resnet18_real_ai_detector",
        }