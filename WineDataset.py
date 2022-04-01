# Regular Imports
import time
import logging
import numpy as np
import pandas as pd

# Accelerators
import cudf
from sklearnex import patch_sklearn

# Data and Metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from cuml.metrics import accuracy_score

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
target = pd.Series(target.astype(np.int32))

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

# --------------------------------------------------------------

# TRAINING PART 3: GPU Accelerated RandomForestClassifer (CUML)

# Convert to gpu based dataframe
X_cudf_train = cudf.DataFrame(X_train)
X_cudf_test = cudf.DataFrame(X_test)

start = time.time()

cuml_acc_sklearn = curfc(
    n_estimators=50,
    max_depth=16,
    random_state=757
)

cuml_acc_sklearn.fit(X_cudf_train, y_train)

stop = time.time()
cuml_acc_time_taken = round(stop-start, 3)

# --------------------------------------------------------------

# INFERENCE 

models = [non_acc_sklearn, acc_sklearn, cuml_acc_sklearn]

inference_times, inference_scores = [], []

for model in models:
    start = time.time()

    predictions = model.predict(X_test)
    score = round(accuracy_score(y_test, predictions), 3)

    stop = time.time()
    total_time = round(stop-start, 3)

    inference_times.append(total_time)
    inference_scores.append(score)

# --------------------------------------------------------------

# Plot training performance
fig = px.bar(
    x = ['Scikit-Learn', 'Scikit-Learn-Intelex', 'RAPIDS cuml'],
    y = [non_acc_time_taken, acc_time_taken, cuml_acc_time_taken],
    text = [non_acc_time_taken, acc_time_taken, cuml_acc_time_taken],
    template = 'plotly_dark',
    color_discrete_sequence = ['#f4a261'],
    title = 'Wine Dataset Training Time with RandomForestClassifier'
)

fig.update_xaxes(title = 'Library Used')
fig.update_yaxes(title = 'Training Time (seconds)')

fig.write_image('plots/WineDataset_TrainingTime_RandomForest.svg')

# --------------------------------------------------------------

# Plot inference performance
df = pd.DataFrame(
    {
        'library': ['Scikit-Learn', 'Scikit-Learn-Intelex', 'RAPIDS cuml'],
        'inference_time': inference_times,
        'inference_score': inference_scores
    }
)

fig1 = px.bar(df, x = 'library', y = 'inference_time', template = 'plotly_dark', 
                    color_discrete_sequence = ['#2a9d8f'], text = 'inference_time',
                    title = 'Wine Dataset Inference Time (seconds) with RandomForestClassifier')

fig2 = px.bar(df, x = 'library', y = 'inference_score', template = 'plotly_dark', 
                    color_discrete_sequence = ['#2a9d8f'], text = 'inference_score',
                    title = 'Wine Dataset Inference Score with RandomForestClassifier')

fig1.write_image('plots/WineDataset_InferenceTime_RandomForest.svg')
fig2.write_image('plots/WineDataset_InferenceScore_RandomForest.svg')

# --------------------------------------------------------------