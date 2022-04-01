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
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

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

# CREATE DATASET

# Parameters for creating some fake data
n_samples = 2**18
n_features = 450
n_info = 300

# Make Data
X,y = make_classification(n_samples=n_samples,
                          n_features=n_features,
                          n_informative=n_info,
                          random_state=757, n_classes=2)
X = pd.DataFrame(X)
y = pd.Series(y.astype(np.float32)) # cuML Random Forest Classifier requires the labels to be integers

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state=757)

# From CPU to GPU
X_cudf_train = cudf.DataFrame.from_pandas(X_train.astype(np.float32))
y_cudf_train = cudf.Series(y_train)
X_cudf_test = cudf.DataFrame.from_pandas((X_test).astype(np.float32))
y_cudf_test = cudf.Series(y_test)

print("X shape: ", X.shape, "\n" +
      "y shape: ", y.shape, "\n")

# --------------------------------------------------------------

# TRAINING PART 1: Non-Accelerated SKLEARN  

print(f'===== Non-Accelerated SKLEARN ===== ')
start = time.time()

non_acc_sklearn = skrfc(
    n_estimators=40,
    max_depth=16,
    random_state=757
)

non_acc_sklearn.fit(X_train, y_train)

stop = time.time()
non_acc_time_taken = round((stop-start)/60, 3)
print(f"Training Time: {non_acc_time_taken} minutes \n")

# --------------------------------------------------------------

# TRAINING PART 2: Accelerated SKLEARN 

print(f'===== Accelerated SKLEARN ===== ')
patch_sklearn()

start = time.time()

acc_sklearn = skrfc(
    n_estimators=40,
    max_depth=16,
    random_state=757
)

acc_sklearn.fit(X_train, y_train)

stop = time.time()
acc_time_taken = round((stop-start)/60, 3)
print(f"Training Time: {acc_time_taken} minutes \n")

# --------------------------------------------------------------

# TRAINING PART 3: GPU Accelerated RandomForestClassifer (CUML)

print(f'===== CUML-Accelerated SKLEARN ===== ')
start = time.time()

cuml_acc_sklearn = curfc(
    n_estimators=40,
    max_depth=16,
    random_state=757,
    max_features=1.0,
    n_streams=1
)

cuml_acc_sklearn.fit(X_cudf_train, y_cudf_train)

stop = time.time()
cuml_acc_time_taken = round((stop-start)/60, 3)
print(f"Training Time: {cuml_acc_time_taken} minutes \n")

# --------------------------------------------------------------

# INFERENCE 

models = [non_acc_sklearn, acc_sklearn, cuml_acc_sklearn]

inference_times, inference_scores = [], []

# SKLEARN MODELS
for model in models:
    start = time.time()

    predictions = model.predict(X_test)
    score = round(accuracy_score(y_test, predictions), 3)

    stop = time.time()
    total_time = round((stop-start)/60, 3)

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
    title = 'Custom Dataset Training Time with RandomForestClassifier'
)

fig.update_xaxes(title = 'Library Used')
fig.update_yaxes(title = 'Training Time (minutes)')

fig.write_image('plots/CustomDataset_TrainingTime_RandomForest.svg')

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
                    title = 'Custom Dataset Inference Time (minutes) with RandomForestClassifier')

fig2 = px.bar(df, x = 'library', y = 'inference_score', template = 'plotly_dark', 
                    color_discrete_sequence = ['#2a9d8f'], text = 'inference_score',
                    title = 'Custom Dataset Inference Score with RandomForestClassifier')

fig1.write_image('plots/CustomDataset_InferenceTime_RandomForest.svg')
fig2.write_image('plots/CustomDataset_InferenceScore_RandomForest.svg')

# --------------------------------------------------------------