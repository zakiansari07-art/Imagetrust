import json
from pathlib import Path

from app.models.forensics_analyzer import ImageForensicsAnalyzer
from app.models.generator_attributor import GeneratorAttributor
from app.models.real_ai_detector import RealAIDetector


DETECTOR_CHECKPOINT = (
    r"C:\Users\Intekhab\Projects\ImageTrust\models\checkpoint"
    r"\resnet_18_ai_detector_best.pth"
)

ATTRIBUTOR_CHECKPOINT = (
    r"C:\Users\Intekhab\Projects\ImageTrust\models\checkpoint"
    r"\gen_model_lr_rate_change_best_validation.pth"
)

TEST_IMAGE_PATH = r"C:\Users\Intekhab\Pictures\test_image.jpeg"


def main():
    if not Path(TEST_IMAGE_PATH).exists():
        raise FileNotFoundError(f"Image not found: {TEST_IMAGE_PATH}")

    detector = RealAIDetector(DETECTOR_CHECKPOINT)
    attributor = GeneratorAttributor(ATTRIBUTOR_CHECKPOINT)
    forensics = ImageForensicsAnalyzer()

    result = {
        "real_vs_ai": detector.predict(TEST_IMAGE_PATH),
        "likely_generator": attributor.predict(TEST_IMAGE_PATH),
        "forensics": forensics.analyze(TEST_IMAGE_PATH),
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()