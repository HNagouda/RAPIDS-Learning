# Imports
import cudf
import numpy as np
import pandas as pd
import pickle

from cuml.ensemble import RandomForestClassifier as curfc
from cuml.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier as skrfc
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import time


# Parameters for creating some fake data
n_samples = 2**15
n_features = 399
n_info = 300
data_type = np.float32


# Make Data
X,y = make_classification(n_samples=n_samples,
                          n_features=n_features,
                          n_informative=n_info,
                          random_state=123, n_classes=2)

X = pd.DataFrame(X.astype(data_type))
# cuML Random Forest Classifier requires the labels to be integers
y = pd.Series(y.astype(np.int32))

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state=0)

# From CPU to GPU
X_cudf_train = cudf.DataFrame.from_pandas(X_train)
X_cudf_test = cudf.DataFrame.from_pandas(X_test)

y_cudf_train = cudf.Series(y_train.values)

print("X shape: ", X.shape, "\n" +
      "y shape: ", y.shape)



print("==== CPU ====", "\n")
# ==== Fit ====

start = time.time()
sk_model = skrfc(n_estimators=40,
                 max_depth=16,
                 max_features=1.0,
                 random_state=10)

sk_model.fit(X_train, y_train)
end = time.time()

print("Training time: {} mins".format(round((end-start)/60, 1)))

# ==== Evaluate ====
start = time.time()
sk_predict = sk_model.predict(X_test)
sk_acc = accuracy_score(y_test, sk_predict)
end = time.time()

print("Evaluation time: {} mins".format(round((end-start)/60, 1)))      


print("==== GPU ====", "\n")
# ==== Fit ====

start = time.time()
cuml_model = curfc(n_estimators=40,
                   max_depth=16,
                   max_features=1.0,
                   random_state=10,
                   n_streams=1) # for reproducibility

cuml_model.fit(X_cudf_train, y_cudf_train)
end = time.time()

print("Training time: {} mins".format(round((end-start)/60, 1)))

# ==== Evaluate ====
start = time.time()
fil_preds_orig = cuml_model.predict(X_cudf_test)

fil_acc_orig = accuracy_score(y_test.to_numpy(), fil_preds_orig)
end = time.time()

print("Evaluation time: {} mins".format(round((end-start)/60, 1)))