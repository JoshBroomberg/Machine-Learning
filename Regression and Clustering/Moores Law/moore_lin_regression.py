import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_validation import train_test_split

### Read Data

def convert_date(full_date_str):
    try:
        date_str = full_date_str.split("-")[1]
        if len(date_str) == 8:
            format_str = "%Y%m%d"
        else:
            format_str = "%y%m%d"

        return int(datetime.strptime(date_str, format_str).strftime("%s"))
    except:
        return 0

type_store = []
def convert_type(type_str):
    return int(type_str.split(".")[0])

benchmarks = np.loadtxt("benchmarks.txt",
    delimiter=",",
    dtype='O',
    converters={
        0: convert_date,
        1: convert_type,
        2: lambda str: float(str)},
    skiprows=1,
    usecols=(0, 1, 2))

# Filter missing dates
benchmarks = benchmarks[np.where(benchmarks[:, 0] > 0)[0]]

# Filter to one benchmark
benchmark = 130.0
selected_benchmarks = benchmarks[np.where(benchmarks[:, 1] == benchmark)[0]]

x_year = selected_benchmarks[:, 0]
y_perf = np.log(selected_benchmarks[:, 2])
plt.scatter(x_year, y_perf)

x_train, x_test, y_train, y_test = train_test_split(x_year, y_perf, test_size=0.1)

# Fit linear model
regr = linear_model.LinearRegression()
regr.fit(x_train.reshape(-1, 1), y_train)

# Run predictions
y_perf_pred = regr.predict(x_test.reshape(-1, 1))

plt.plot(x_test, y_perf_pred, color="r")
plt.show()

# The coefficients
print('Coefficients: \n', regr.coef_)

# The mean squared error
print("Mean squared error: %.2f"
            % mean_squared_error(y_test, y_perf_pred))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_perf_pred))