import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def scale_train(X_train, X_prep_train):
    # Fit Scale X Variables
    scaler = StandardScaler()
    X_scaler = scaler.fit(X_prep_train)
    
    # Scale X Variables
    X_train_scaled = X_scaler.transform(X_train)
    
    # Scale based on volatility
    X0_train_scaled = X_scaler.transform(X_train[X_prep_train['labels'] == 0])
    X1_train_scaled = X_scaler.transform(X_train[X_prep_train['labels'] == 1])
    X2_train_scaled = X_scaler.transform(X_train[X_prep_train['labels'] == 2])
    
    y_train = X_prep_train['y'].values
    y0_train = X_prep_train[X_prep_train['labels'] == 0]['y'].values
    y1_train = X_prep_train[X_prep_train['labels'] == 1]['y'].values
    y2_train = X_prep_train[X_prep_train['labels'] == 2]['y'].values
    
    # Convert to arrays
    X_train_scaled = np.array(X_train_scaled)
    X0_train_scaled = np.array(X0_train_scaled)
    X1_train_scaled = np.array(X1_train_scaled)
    X2_train_scaled = np.array(X2_train_scaled)
    y_train = np.array(y_train)
    y0_train = np.array(y0_train)
    y1_train = np.array(y1_train)
    y2_train = np.array(y2_train)
    
    return X_train_scaled, X0_train_scaled, X1_train_scaled, X2_train_scaled, y_train, y0_train, y1_train, y2_train
