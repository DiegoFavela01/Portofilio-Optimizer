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
import traceback

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

# Pull Data from APIs and divide into test and train datasets
try:
    X_train_scaled, y_train, X_test_scaled, y_test, X_prep_train, X_prep_test = create_train_test_tables()
    st.write("Data has been prepared")

    # Print shapes of X_train and X_test for debugging
    st.write("X_train shape:", X_train_scaled.shape)
    st.write("X_test shape:", X_test_scaled.shape)

    # Scale the training and test data
    X_train_scaled, X0_train_scaled, X1_train_scaled, X2_train_scaled, y_train, y0_train, y1_train, y2_train = scale_train(X_train_scaled, X_prep_train)
    X_test_scaled, X0_test_scaled, X1_test_scaled, X2_test_scaled, y_test, y0_test, y1_test, y2_test = scale_test(X_train_scaled, X_test_scaled, X_prep_test)
    st.write("Data has been scaled")
    st.write("X_train_scaled shape:", X_train_scaled.shape)
    st.write("X_train_scaled dtype:", X_train_scaled.dtype)
    st.write("y_train dtype:", y_train.dtype)

    # Train the regression model
    try:
        regression_model = Sequential()
        input_dim = X_train_scaled.shape[1]  # Use the second dimension of X_train_scaled to determine input dimension
        regression_model.add(Dense(32, activation='relu', input_dim=input_dim))
        regression_model.add(Dense(1))
        regression_model.compile(optimizer=Adam(), loss='mse')
        X_train_scaled = X_train_scaled.astype(np.float32)
        y_train = y_train.astype(np.float32)
        regression_model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=0)
        st.write("Regression model training complete")
    except Exception as e:
        st.error("Error occurred while training the regression model.")
        st.error(traceback.format_exc())

    # Train the classification model
    try:
        classification_model = Sequential()
        classification_model.add(Dense(32, activation='relu', input_dim=X_train_scaled.shape[1]))
        classification_model.add(Dense(1, activation='sigmoid'))
        classification_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        classification_model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=0)
        st.write("Classification model training complete")
    except Exception as e:
        st.error("Error occurred while training the classification model.")
        st.error(traceback.format_exc())

    # Predict the returns using the regression model
    try:
        strategy_returns_reg = regression_model.predict(X_test_scaled).flatten()
        st.write("Regression predictions generated")
    except Exception as e:
        st.error("Error occurred while generating regression predictions.")
        st.error(traceback.format_exc())

    # Predict the returns using the classification model
    try:
        strategy_returns_class = classification_model.predict(X_test_scaled).flatten()
        st.write("Classification predictions generated")
    except Exception as e:
        st.error("Error occurred while generating classification predictions.")
        st.error(traceback.format_exc())

    # Prepare returns DataFrame
    returns_data = {
        'Date': X_prep_test.index,
        'Strategy Returns (Regression)': strategy_returns_reg,
        'Strategy Returns (Classification)': strategy_returns_class,
        'Benchmark Returns (S&P 500)': y_test
    }
    returns_df = pd.DataFrame(data=returns_data, index=X_prep_test.index)

    # Generate Pyfolio analysis for regression strategy
    try:
        regression_returns = returns_df['Strategy Returns (Regression)']
        if regression_returns.empty:
            st.warning("No data available for the regression strategy.")
        else:
            st.write(f"Regression Returns Shape: {regression_returns.shape}")
            regression_tear_sheet = pf.create_returns_tear_sheet(
                returns=regression_returns,
                benchmark_rets=returns_df['Benchmark Returns (S&P 500)'],
                return_fig=True
            )

            # Streamlit app
            st.title("Machine Learning Strategy Evaluation")
            st.write("## Performance Analysis - Regression Strategy")

            # Display Pyfolio returns tear sheet for regression strategy
            st.pyplot(regression_tear_sheet)

            # Display raw data if desired
            if st.checkbox("Show Data - Regression Strategy"):
                st.write(returns_df[['Strategy Returns (Regression)', 'Benchmark Returns (S&P 500)']])
    except Exception as e:
        st.error("An error occurred while generating the Pyfolio analysis for the regression strategy.")
        st.error(traceback.format_exc())

    # Generate Pyfolio analysis for classification strategy
    try:
        if returns_df["Strategy Returns (Classification)"].empty:
            st.warning("No data available for the classification strategy.")
        else:
            classification_tear_sheet = pf.create_returns_tear_sheet(
                returns=returns_df['Strategy Returns (Classification)'],
                benchmark_rets=returns_df['Benchmark Returns (S&P 500)'],
                return_fig=True
            )

            # Streamlit app
            st.title("Machine Learning Strategy Evaluation")
            st.write("## Performance Analysis - Classification Strategy")

            # Display Pyfolio returns tear sheet for classification strategy
            st.pyplot(classification_tear_sheet)

            # Display raw data if desired
            if st.checkbox("Show Data - Classification Strategy"):
                st.write(returns_df[['Strategy Returns (Classification)', 'Benchmark Returns (S&P 500)']])
    except Exception as e:
        st.error("An error occurred while generating the Pyfolio analysis for the classification strategy.")
        st.error(traceback.format_exc())

except Exception as e:
    st.error("An error occurred while running the application.")
    st.error(traceback.format_exc())
