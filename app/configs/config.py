from pathlib import Path
import os
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DETECTOR_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "checkpoint"
    / "resnet_18_ai_detector_best.pth"
)

ATTRIBUTOR_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "checkpoint"
    / "gen_model_lr_rate_change_best_validation.pth"
)


DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing. Add it to your .env file.")

