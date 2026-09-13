from app.database.database import Base, engine
from app.database.db_models import AnalysisRecord  


def main():
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")


if __name__ == "__main__":
    main()