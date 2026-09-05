from app.database import Base, engine
from app.db_models import AnalysisRecord  # Registers the table with Base.


def main():
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")


if __name__ == "__main__":
    main()