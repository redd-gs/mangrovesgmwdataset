import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

class DatabaseConnection:
    def __init__(self):
        self.host = os.environ.get("PGHOST", "localhost")
        self.port = os.environ.get("PGPORT", "5432")
        self.database = os.environ.get("PGDATABASE", "global_mangrove_dataset")
        self.user = os.environ.get("PGUSER", "postgres")
        self.password = os.environ.get("PGPASSWORD", "mangrovesondra")  # Update with your password

        self.engine = self.create_engine()
        self.Session = sessionmaker(bind=self.engine)

    def create_engine(self):
        connection_string = f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        return create_engine(connection_string)

    def get_session(self):
        return self.Session()