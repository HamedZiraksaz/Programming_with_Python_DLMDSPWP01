import pandas as pd
from errors import DataLoadError


class CSVReader:
    # loads csv file into a DataFrame

    def load_csv(self, path):
        """reads a csv file, turns any loading problem into DataLoadError"""
        try:
            data = pd.read_csv(path)
            return data
        except FileNotFoundError:
            raise DataLoadError("File not found: " + path)
        except Exception as e:
            raise DataLoadError("Could not read file " + path + ": " + str(e))
