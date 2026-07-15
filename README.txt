DLMDSPWP01 -  "Programming with Python" .

It picks the best-matching ideal function (out of 50) for each of 4 training
columns using least squares, then checks each test point against the
chosen functions using the sqrt(2) deviation rule

Files

main.py - runs everything
csv_reader.py - loads CSV files with pandas
database.py - saves/loads tables in a SQLite database
calculations.py - the selection and mapping logic
charts.py- draws the charts with Bokeh
errors.py custom exceptions (DataLoadError, TooFarError)
test_file.py unit tests (pytest)

Setup:
pip install -r requirements.txt


Put: train.csv, ideal.csv and test.csv in the data folder.

Run: 
python main.py

Test:
pytest test_file.py -v

