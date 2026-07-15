from sqlalchemy import create_engine
import pandas as pd
from csv_reader import CSVReader


class Database:
    # handles saving/loading with SQLite via SQLAlchemy.

    def __init__(self, db_path):
        self.engine = create_engine("sqlite:///" + db_path)
        self.reader = CSVReader()

    def load_csv(self, path):
        # just uses CSVReader, doesn't load csv itself
        return self.reader.load_csv(path)

    def save_table(self, df, name):
        # writes a DataFrame to a table, replaces it if it already exists
        df.to_sql(name, self.engine, if_exists='replace', index=False)

    def read_table(self, name):
        # reads a table back as a DataFrame
        query = "SELECT * FROM " + name
        return pd.read_sql(query, self.engine)
