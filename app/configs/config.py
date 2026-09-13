from pathlib import Path
import os
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
print(PROJECT_ROOT)

load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DETECTOR_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_models_checkpoint"
    / "bin_task_defactify_best_validation.pth"
)

ATTRIBUTOR_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_models_checkpoint"
    / "multi_task_defactify_last_checkpoint.pth"
)


DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing. Add it to your .env file.")

