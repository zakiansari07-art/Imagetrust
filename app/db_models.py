from sqlalchemy import DateTime, String, func, Float
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import JSONB
from app.database import Base
from datetime import datetime



class AnalysisRecord(Base):
    __tablename__ = "new_record"

    id : Mapped[int] = mapped_column(primary_key= True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    image_file_name: Mapped[str] = mapped_column(String(255), nullable=False)

    verdict : Mapped[str] = mapped_column(String(50), nullable=False)

    confidence : Mapped[float | None] = mapped_column(Float, nullable=True)

    confidence_level : Mapped[str] = mapped_column(String(50), nullable=False)

    likely_generator: Mapped[str | None] = mapped_column(String(50), nullable=True)

    generator_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    result_json : Mapped[dict] = mapped_column(JSONB, nullable=False)
