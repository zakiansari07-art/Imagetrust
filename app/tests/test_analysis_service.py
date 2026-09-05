import json
from pathlib import Path

from app.services.analysis_service import AnalysisService


DETECTOR_CHECKPOINT = (
    r"C:\Users\Intekhab\Projects\ImageTrust\models\checkpoint"
    r"\resnet_18_ai_detector_best.pth"
)

ATTRIBUTOR_CHECKPOINT = (
    r"C:\Users\Intekhab\Projects\ImageTrust\models\checkpoint"
    r"\gen_model_lr_rate_change_best_validation.pth"
)

TEST_IMAGE_PATH = r"C:\Users\Intekhab\Pictures\test_image.png"


def main():
    if not Path(TEST_IMAGE_PATH).exists():
        raise FileNotFoundError(f"Image not found: {TEST_IMAGE_PATH}")

    service = AnalysisService(
        detector_checkpoint_path=DETECTOR_CHECKPOINT,
        attributor_checkpoint_path=ATTRIBUTOR_CHECKPOINT,
    )

    result = service.analyze(TEST_IMAGE_PATH)

    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()