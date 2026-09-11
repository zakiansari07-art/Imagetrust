import shutil
import traceback
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi.responses import JSONResponse

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from app.configs.config import (
    ATTRIBUTOR_CHECKPOINT_PATH,
    DETECTOR_CHECKPOINT_PATH,
)
from app.schemas.predictions import AnalysisResult
from app.services.analysis_service import AnalysisService
from app.repositories.analysis_repository import AnalysisRepository
from app.database.database import SessionLocal
from app.frontend.gradio_ui import demo
import gradio as gr
from sqlalchemy.exc import SQLAlchemyError
import logging

logger = logging.getLogger(__name__)



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Models load once when the API starts—not for every uploaded image.
    app.state.analysis_service = AnalysisService(
        detector_checkpoint_path=str(DETECTOR_CHECKPOINT_PATH),
        attributor_checkpoint_path=str(ATTRIBUTOR_CHECKPOINT_PATH),

  
    )
    yield


app = FastAPI(
    title="ImageTrust API",
    description="Analyze images for likely AI generation and source attribution.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception("UNHANDLED ERROR on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalysisResult)
def analyze_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Please upload an image file.",
        )

    suffix = Path(file.filename or "upload.png").suffix
    temp_path = None
    db = SessionLocal()

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temporary_file:
            shutil.copyfileobj(file.file, temporary_file)
            temp_path = Path(temporary_file.name)

       
        result = app.state.analysis_service.analyze(str(temp_path))

        repository = AnalysisRepository(db=db)
        repository.save(image_file_name=Path(file.filename or "upload").name, result=result)

        return result




    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be read as an image.",
        ) from error

    except SQLAlchemyError as error:
        db.rollback()
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {error}",
        ) from error

    except Exception as error:
        logger.exception("ANALYZE FAILED")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {error}",
        ) from error

    finally:
        db.close()
        file.file.close()

        if temp_path and temp_path.exists():
            temp_path.unlink()

app = gr.mount_gradio_app(app, demo, path="/")