# Regular Imports
from re import template
import time
import logging
import pandas as pd

# Accelerators
import cudf
from sklearnex import patch_sklearn

# Data and Metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from cuml.metrics import r2_score

# Models
from sklearn.ensemble import RandomForestClassifier as skrfc
from cuml.ensemble import RandomForestClassifier as curfc

# Visualization
import plotly.express as px

# --------------------------------------------------------------

# Create Logger
logger = logging.getLogger()
fh = logging.FileHandler('RandomForestLog.txt')

fh.setLevel(10)
logger.addHandler(fh)

# --------------------------------------------------------------

# Load Dataset
wine = load_wine()
features, target = load_wine(return_X_y=True)

# Create splits
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.25, random_state=757
)

# --------------------------------------------------------------

# TRAINING PART 1: Non-Accelerated SKLEARN
start = time.time()

non_acc_sklearn = skrfc(
    n_estimators=50,
    max_depth=16,
    random_state=757
)

non_acc_sklearn.fit(X_train, y_train)

stop = time.time()
non_acc_time_taken = round(stop-start, 3)

# logger.log(f'Non-Accelerated SKLEARN RandomForestClassifier training time: {non_acc_time_taken} seconds')

# --------------------------------------------------------------

# TRAINING PART 2: Accelerated SKLEARN
patch_sklearn()

start = time.time()

acc_sklearn = skrfc(
    n_estimators=50,
    max_depth=16,
    random_state=757
)

acc_sklearn.fit(X_train, y_train)

stop = time.time()
acc_time_taken = round(stop-start, 3)

# logger.log(f'Non-Accelerated SKLEARN RandomForestClassifier training time: {acc_time_taken} seconds')

# --------------------------------------------------------------

# TRAINING PART 3: GPU Accelerated RandomForestClassifer (CUML)
cu_X_train = cudf.DataFrame(X_train)
cu_y_train = cudf.DataFrame(y_train)

start = time.time()

cuml_acc_sklearn = curfc(
    n_estimators=50,
    max_depth=16,
    random_state=757
)

cuml_acc_sklearn.fit(cu_X_train, cu_y_train)

stop = time.time()
cuml_acc_time_taken = round(stop-start, 3)

# logger.log(f'Non-Accelerated SKLEARN RandomForestClassifier training time: {cuml_acc_time_taken} seconds')

# --------------------------------------------------------------

# Plot performance
fig = px.bar(
    x = ['Scikit-Learn', 'Scikit-Learn-Intelex', 'RAPIDS cuml'],
    y = [non_acc_time_taken, acc_time_taken, cuml_acc_time_taken],
    text = [non_acc_time_taken, acc_time_taken, cuml_acc_time_taken],
    template = 'plotly_dark',
    color_discrete_sequence = ['#f4a261'],
    title = 'Wine Dataset Training Time'
)

fig.update_xaxes(title = 'Library Used')
fig.update_yaxes(title = 'Training Time (seconds)')

fig.write_image('plots/RandomForestPerformance.png')
fig.write_image('plots/RandomForestPerformance.svg')