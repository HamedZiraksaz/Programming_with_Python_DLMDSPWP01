"""
load train.csv, ideal.csv, test.csv, save raw data into a SQLite database, pick the best ideal function per training column, map test points to chosen functions and draw the charts
"""

import os
import sys
import pandas as pd

from database import Database
from errors import DataLoadError
from calculations import Calculator
from charts import Charts

DB_PATH = "saved_files/my_database.db"


def load_test_data_line_by_line(path):
    # reads test.csv line by line, as required
    rows = []
    with open(path, "r") as f:
        f.readline()  # skip header
        for line in f:
            line = line.strip()
            if line == "":
                continue
            parts = line.split(",")
            rows.append({"x": float(parts[0]), "y": float(parts[1])})
    return pd.DataFrame(rows)


def main():
    # runs the whole pipeline: load, save, pick, map, save, draw, print summary
    os.makedirs('saved_files', exist_ok=True)
    db = Database(DB_PATH)

    try:
        train = db.load_csv("data/train.csv")
        ideal = db.load_csv("data/ideal.csv")
        test = load_test_data_line_by_line("data/test.csv")
    except DataLoadError as e:
        print("Could not load a file:", e)
        sys.exit(1)
    except FileNotFoundError as e:
        print("File not found:", e)
        sys.exit(1)

    db.save_table(train, "training_data")
    db.save_table(ideal, "ideal_functions")
    db.save_table(test, "test_data_raw")

    calc = Calculator()
    picks = calc.pick_ideals_for_training(train, ideal)
    mapped_df = calc.map_test_points(test, ideal, picks)

    mapped_only = mapped_df[mapped_df["status"] == "mapped"].copy()
    results = mapped_only[["x", "y", "delta_y", "ideal_no"]].copy()
    results = results.rename(columns={"x": "x_test", "y": "y_test", "ideal_no": "ideal_func_no"})
    db.save_table(results, "test_mapping_results")

    charts = Charts()
    charts.draw_all_charts(train, ideal, picks, mapped_df, output_path="charts.html", auto_open=True)

    print("Selected ideal functions:")
    for t in range(1, 5):
        p = picks[t]
        print("Y" + str(t), "-> ideal function", p["ideal_no"],
              "| SSE =", round(p["sse"], 2),
              "| max deviation =", round(p["max_dev"], 3))

    print("Done. Charts saved to charts.html and results saved to the database.")


if __name__ == "__main__":
    main()
