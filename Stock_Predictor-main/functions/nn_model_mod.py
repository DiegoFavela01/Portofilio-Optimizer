# General packages
import pandas as pd
import numpy as np

# Packages related to machine learning
#for nueral networs
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.models import load_model

# fix random seed for same reproducibility as my results due to stochastic nature of start point
K.clear_session()
tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

def nn_reg_model (X_train_scaled, y_train):
    # Set Training epoch end limits, save model with the best fit during epoch testing.
    call = [tf.keras.callbacks.EarlyStopping(monitor='loss', 
                                                  mode='min', 
                                                  patience=35, 
                                                  verbose=1,
                                                 ),
                 tf.keras.callbacks.ModelCheckpoint(filepath='best_nn_model.h5', 
                                                    monitor='loss', 
                                                    mode='min',
                                                    save_best_only=True, 
                                                    initil_value_threshold = .04
                                                    )
                ]
    # create a loop to ensure that the fit of the machine learning model meets certain requirements
    i=10
    b=10
    while (i >= 1.39) or (b >= .05):
        # fix random seed for same reproducibility as my results due to stochastic nature of start point
        K.clear_session()
        tf.keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)

        # Create nueral network
        nn = Sequential()

        # add input layer
        nn.add(Dense(units=100, input_dim=49, activation="relu"))
        # add first hidden layer
        nn.add(Dense(units=150, activation="relu"))
        # add third hidden layer
        nn.add(Dense(units=10, activation="relu"))
        # Output layer
        nn.add(Dense(units=1, activation="linear"))
        # Compile the model
        nn.compile(loss="mean_squared_error", optimizer='adam', metrics=['mean_squared_error'])
        try:
            # Fit the model
            nn_model = nn.fit(X_train_scaled, y_train, validation_split = 0.2, epochs=300, batch_size=64, callbacks = call, verbose=0)
            b = pd.DataFrame(nn_model.history['loss']).min().values
            i = nn_model.history['val_loss'][-1]
        except:
            # Fit the model
            nn_model = nn.fit(X_train_scaled, y_train, validation_split = 0.2, epochs=300, batch_size=64, callbacks = call, verbose=0)
            b = pd.DataFrame(nn_model.history['loss']).min().values
            i = nn_model.history['val_loss'][-1]

    # load a saved model
    saved_nn_model = load_model('best_nn_model.h5')
    
    return saved_nn_model, nn_model