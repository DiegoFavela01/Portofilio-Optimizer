import pandas as pd
import datetime as dt
from sklearn.preprocessing import StandardScaler
import numpy as np
np.random.seed(42)

def scale_test(X_train, X_test, X_prep_test):
    # Fit scaler on X_train
    scaler = StandardScaler()
    X_scaler = scaler.fit(X_train.astype(np.float32))

    # Scale X_test
    X_test_scaled = X_scaler.transform(X_test.astype(np.float32))

    # Subset variables based on labels
    X0_test_scaled = X_scaler.transform(X_test[X_prep_test['labels'] == 0].astype(np.float32))
    X1_test_scaled = X_scaler.transform(X_test[X_prep_test['labels'] == 1].astype(np.float32))
    X2_test_scaled = X_scaler.transform(X_test[X_prep_test['labels'] == 2].astype(np.float32))

    y_test = X_prep_test['y'].values
    y0_test = X_prep_test[X_prep_test['labels'] == 0]['y'].values
    y1_test = X_prep_test[X_prep_test['labels'] == 1]['y'].values
    y2_test = X_prep_test[X_prep_test['labels'] == 2]['y'].values

    return (
        X_test_scaled,
        X0_test_scaled,
        X1_test_scaled,
        X2_test_scaled,
        y_test,
        y0_test,
        y1_test,
        y2_test,
    )
