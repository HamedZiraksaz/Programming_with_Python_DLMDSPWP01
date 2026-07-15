import os
import webbrowser
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column


class Charts:
    # build the plots

    def draw_training_plot(self, x_values, y_values, title, color):
        # one scatter plot for one training column
        plot = figure(title=title, x_axis_label="X", y_axis_label="Y", width=900, height=300)
        plot.scatter(x_values, y_values, size=5, color=color, alpha=0.8)
        return plot

    def draw_overlay_plot(self, train_df, ideal_df, picks, mapped_df):
        # one combined chart: all 4 training columns, their ideal functions, and test points
        colors = {1: "blue", 2: "green", 3: "orange", 4: "purple"}
        x_train = train_df.iloc[:, 0]

        plot = figure(
            title="Training data, chosen ideal functions and test points",
            x_axis_label="X",
            y_axis_label="Y",
            width=920,
            height=550,
        )

        for t in range(1, 5):
            y_train = train_df.iloc[:, t]
            color = colors[t]
            plot.scatter(x_train, y_train, size=5, color=color, alpha=0.5,
                         legend_label="Training Y" + str(t))

            ideal_no = picks[t]["ideal_no"]
            y_ideal = ideal_df.iloc[:, ideal_no]
            plot.line(x_train, y_ideal, line_width=2, color=color,
                      legend_label="Ideal function " + str(ideal_no))

        # mapped points get a black triangle, rejected points a small grey cross
        for _, row in mapped_df.iterrows():
            if row["status"] == "mapped":
                plot.scatter(row["x"], row["y"], size=10, color="black", marker="triangle")
            else:
                plot.scatter(row["x"], row["y"], size=8, color="gray", marker="x")

        plot.legend.location = "top_left"
        plot.legend.click_policy = "hide"
        return plot

    def draw_all_charts(self, train_df, ideal_df, picks, mapped_df,
                         output_path="charts.html", auto_open=True):
        # stacks all 4 single plots + the combined plot into one html file
        colors = {1: "blue", 2: "green", 3: "orange", 4: "purple"}
        x_train = train_df.iloc[:, 0]

        plots = []
        for t in range(1, 5):
            y_train = train_df.iloc[:, t]
            title = "Training Function Y" + str(t)
            plots.append(self.draw_training_plot(x_train, y_train, title, colors[t]))

        plots.append(self.draw_overlay_plot(train_df, ideal_df, picks, mapped_df))

        output_file(output_path, title="Charts")
        save(column(*plots))

        if auto_open:
            webbrowser.open("file://" + os.path.abspath(output_path))
