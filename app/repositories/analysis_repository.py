from app.schemas.predictions import AnalysisResult
from app.db_models import AnalysisRecord
from sqlalchemy.orm import Session


class AnalysisRepository:
    def __init__(self, db: Session):
        self.db = db

    def save(self, image_file_name: str, result: AnalysisResult)-> AnalysisRecord:

        record = AnalysisRecord(
            image_file_name=image_file_name,
            verdict=result.verdict,
            confidence=result.confidence,
            source_status = result.source_status,
            likely_generator = result.likely_generator,
            generator_confidence=  result.generator_confidence,
            result_json = result.model_dump(mode="json")
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)

        return record