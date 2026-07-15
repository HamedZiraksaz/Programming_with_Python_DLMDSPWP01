# tests for Calculator + CSVReader


import numpy as np
import pandas as pd

from calculations import Calculator
from errors import TooFarError, DataLoadError
from csv_reader import CSVReader


def test_threshold_violation():
    # a deviation of 10 against a max training deviation of 1 should be rejected
    calc = Calculator()
    got_error = False
    try:
        calc.check_test_point(y_test=10.0, y_ideal=0.0, max_train_dev=1.0)
    except TooFarError:
        got_error = True
    assert got_error


def test_find_best_ideal_exact_match():
    # candidate 2 is an exact copy of train_y, so it should win with SSE = 0
    calc = Calculator()
    train_y = np.array([1.0, 2.0, 3.0])
    ideal_ys = {1: np.array([0.0, 0.0, 0.0]), 2: np.array([1.0, 2.0, 3.0])}
    best_id, sse = calc.find_best_ideal(train_y, ideal_ys)
    assert best_id == 2
    assert sse == 0.0


def test_find_max_deviation():
    # biggest gap between [0,1,2] and [0,0,0] is 2
    calc = Calculator()
    result = calc.find_max_deviation(np.array([0., 1., 2.]), np.array([0., 0., 0.]))
    assert result == 2.0


def test_load_csv_missing_file():
    # a missing file should raise our own DataLoadError, not pandas' error
    reader = CSVReader()
    got_error = False
    try:
        reader.load_csv("no_such_file.csv")
    except DataLoadError:
        got_error = True
    assert got_error


def test_map_test_points():
    # 3 test points, one for each possible outcome: mapped, rejected, x_not_found
    calc = Calculator()
    ideal_df = pd.DataFrame({"x": [0.0, 1.0], "y1": [0.0, 1.0], "y2": [10.0, 10.0]})
    picks = {
        1: {"ideal_no": 1, "sse": 0.0, "max_dev": 0.1},
        2: {"ideal_no": 2, "sse": 0.0, "max_dev": 0.1},
        3: {"ideal_no": 1, "sse": 0.0, "max_dev": 0.1},
        4: {"ideal_no": 1, "sse": 0.0, "max_dev": 0.1},
    }
    test_df = pd.DataFrame({"x": [0.0, 1.0, 5.0], "y": [0.05, 100.0, 0.0]})
    result = calc.map_test_points(test_df, ideal_df, picks)

    assert result.iloc[0]["status"] == "mapped"
    assert result.iloc[0]["ideal_no"] == 1
    assert result.iloc[1]["status"] == "rejected"
    assert result.iloc[2]["status"] == "x_not_found"

