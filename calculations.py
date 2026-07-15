import numpy as np
import pandas as pd
from errors import TooFarError


class Calculator:
    # picks the best ideal function, maps every test point to one of the chosen functions.

    def make_ideal_dict(self, ideal_df):
        # turns the 50-column ideal dataframe into {function number: array of values}
        ideal_dict = {}
        n_cols = ideal_df.shape[1]
        for i in range(1, n_cols):
            ideal_dict[i] = ideal_df.iloc[:, i].to_numpy(dtype=float)
        return ideal_dict

    def find_best_ideal(self, train_y, ideal_ys):
        # compares train_y against every candidate, keeps the smallest SSE
        best_id = None
        best_sse = None
        for func_id, ideal_y in ideal_ys.items():
            difference = train_y - ideal_y
            sse = float(np.sum(difference ** 2))
            if best_sse is None or sse < best_sse:
                best_sse = sse
                best_id = func_id
        return best_id, best_sse

    def find_max_deviation(self, train_y, ideal_y):
        # biggest single absolute difference between two arrays
        diffs = np.abs(train_y - ideal_y)
        return float(np.max(diffs))

    def check_test_point(self, y_test, y_ideal, max_train_dev):
        # raises TooFarError if the point is further than sqrt(2) * max_train_dev
        deviation = abs(y_test - y_ideal)
        limit = (2 ** 0.5) * max_train_dev
        if deviation > limit:
            raise TooFarError(
                "deviation " + str(deviation) + " is bigger than limit " + str(limit)
            )
        return deviation

    def pick_ideals_for_training(self, train_df, ideal_df):
        # loops over all 4 training columns, picks the best ideal function for each
        ideal_dict = self.make_ideal_dict(ideal_df)
        picks = {}
        for t in range(1, 5):
            train_y = train_df.iloc[:, t].to_numpy(dtype=float)
            best_id, sse = self.find_best_ideal(train_y, ideal_dict)
            max_dev = self.find_max_deviation(train_y, ideal_dict[best_id])
            picks[t] = {"ideal_no": best_id, "sse": sse, "max_dev": max_dev}
        return picks

    def find_row_for_x(self, x, ideal_df):
        # finds the row where the x-value matches, or None if not found
        matches = ideal_df.index[ideal_df.iloc[:, 0] == x].tolist()
        if len(matches) == 0:
            return None
        return matches[0]

    def map_test_points(self, test_df, ideal_df, picks):
        # checks every test point against the 4 chosen functions, keeps the closest match
        rows = []
        for _, test_row in test_df.iterrows():
            x = float(test_row.iloc[0])
            y = float(test_row.iloc[1])
            row_idx = self.find_row_for_x(x, ideal_df)

            best_delta = None
            best_t = None
            best_ideal_no = None

            if row_idx is not None:
                for t in range(1, 5):
                    ideal_no = picks[t]["ideal_no"]
                    max_dev = picks[t]["max_dev"]
                    y_ideal = float(ideal_df.iloc[row_idx, ideal_no])
                    try:
                        delta = self.check_test_point(y, y_ideal, max_dev)
                    except TooFarError:
                        continue  # doesn't fit this one, try the next
                    if best_delta is None or delta < best_delta:
                        best_delta = delta
                        best_t = t
                        best_ideal_no = ideal_no

            if best_delta is None:
                status = "rejected" if row_idx is not None else "x_not_found"
                rows.append({
                    "x": x, "y": y, "status": status,
                    "training_idx": None, "ideal_no": None, "delta_y": None,
                })
            else:
                rows.append({
                    "x": x, "y": y, "status": "mapped",
                    "training_idx": best_t, "ideal_no": best_ideal_no, "delta_y": best_delta,
                })

        return pd.DataFrame(rows)
