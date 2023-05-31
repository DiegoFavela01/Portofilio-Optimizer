import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import pyfolio as pf
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Turn off warning signs for cleaner code
from warnings import filterwarnings
filterwarnings("ignore")

# Import modules
from functions.vix_mod import vix_analysis
from functions.spy_mod import spy_analysis
from functions.econ_mod import get_econ_data
from functions.sent_mod import market_sent
from functions.create_train_test_mod import create_train_test_tables
from functions.test_scaled_mod import scale_test
from functions.train_scaled_mod import scale_train
from functions.nn_model_mod import nn_reg_model
from functions.nn_class_model_mod import nn_class_model
from functions.test_scaled_mod import scale_test
from functions.create_train_test_mod import create_train_test_tables
from functions.train_scaled_mod import scale_train
from functions.test_scaled_mod import scale_test

# Pull Data from APIs and divide into test and train datasets
X_train, y_train, X_test, y_test, X_prep_train, X_prep_test = create_train_test_tables()
st.write("Data has been prepared")

# Print shapes of X_train and X_test for debugging
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Scale the training and test data
X_train_scaled = scale_train(X_train, X_prep_train)
X_test_scaled = scale_test(X_train, X_test, X_prep_test)
st.write("Data has been scaled")
print("X_train_scaled shape:", X_train_scaled.shape)

# Scale the training and test data
X_train_scaled, X_test_scaled = scale_train(X_train, X_prep_train), scale_test(X_train, X_test, X_prep_test)
X_train_scaled = np.array(X_train_scaled)
X_test_scaled = np.array(X_test_scaled)
st.write("Data has been scaled")
print("X_train_scaled shape:", X_train_scaled.shape)

# Train the regression model
regression_model = Sequential()
input_dim = X_train_scaled.shape[1]  # Use the second dimension of X_train_scaled to determine input dimension
regression_model.add(Dense(32, activation='relu', input_dim=input_dim))
regression_model.add(Dense(1))
regression_model.compile(optimizer=Adam(), loss='mse')

# Convert X_train_scaled and y_train to compatible types if needed
if not isinstance(X_train_scaled, np.ndarray):
    X_train_scaled = np.array(X_train_scaled)
if not isinstance(y_train, np.ndarray):
    y_train = np.array(y_train)

# Check the data types of X_train_scaled and y_train
print(X_train_scaled.dtype, y_train.dtype)

# Convert the data types if needed
X_train_scaled = X_train_scaled.astype(np.float32)
y_train = y_train.astype(np.float32)


# Fit the regression model
regression_model.fit(tf.convert_to_tensor(X_train_scaled), tf.convert_to_tensor(y_train), epochs=10, batch_size=32, verbose=0)
st.write("Regression model training complete")

# Train the classification model
classification_model = Sequential()
classification_model.add(Dense(32, activation='relu', input_dim=X_train_scaled.shape[1]))
classification_model.add(Dense(1, activation='sigmoid'))
classification_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
classification_model.fit(X_train_scaled, X_prep_train, epochs=10, batch_size=32, verbose=0)
st.write("Classification model training complete")

# Predict the returns using the regression model
strategy_returns_reg = regression_model.predict(X_test_scaled).flatten()

# Predict the returns using the classification model
strategy_returns_class = classification_model.predict(X_test_scaled).flatten()

# Prepare benchmark returns (S&P 500)
benchmark_returns = y_test.flatten()

# Create Pyfolio returns DataFrame
returns_df = pd.DataFrame({
    'Strategy Returns (Regression)': strategy_returns_reg,
    'Strategy Returns (Classification)': strategy_returns_class,
    'Benchmark Returns (S&P 500)': benchmark_returns
}, index=X_test.index)

# Generate Pyfolio analysis
tear_sheet = pf.create_returns_tear_sheet(returns=returns_df, return_fig=True)

# Streamlit app
st.title("Machine Learning Strategy Evaluation")
st.write("## Performance Analysis")

# Display Pyfolio returns tear sheet
st.pyplot(tear_sheet)

# Display raw data if desired
if st.checkbox("Show Data"):
    st.write(returns_df)