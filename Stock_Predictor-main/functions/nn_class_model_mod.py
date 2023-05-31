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

def nn_class_model (X_train_scaled, X_prep_train):
# establish callback functions
    call = [tf.keras.callbacks.EarlyStopping(monitor="binary_crossentropy", 
                                          mode='min', 
                                          patience=35, 
                                          ),
         tf.keras.callbacks.ModelCheckpoint(filepath='best_nn_class_model.h5', 
                                            monitor="val_accuracy", 
                                            mode='max',
                                            save_best_only=True, 
                                            )
        ]

# Convert y-train to binary
    y_train = pd.DataFrame(np.where(X_prep_train['y']>=0,1,-1))
# create a loop to ensure that the fit of the machine learning model meets certain requirements
    i=0
    b=0 
# loop model until a desirable fit has been returned
    while (i <= .50) or (b <= .22):
        # fix random seed for same reproducibility as my results due to stochastic nature of start point
        K.clear_session()
        tf.keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)

        # Create nueral network
        nn_class = Sequential()
        # add first hidden layer
        nn_class.add(Dense(units=150, input_dim=49, activation="relu"))
        # add second hidden layer
        nn_class.add(Dense(units=150, activation="relu"))
        # add third hidden layer
        nn_class.add(Dense(units=100, activation="relu"))
        # add fourth hidden layer
        nn_class.add(Dense(units=5, activation="sigmoid"))
        # Output layer
        nn_class.add(Dense(units=1, activation="sigmoid"))

        # Compile the model
        nn_class.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['accuracy', "binary_crossentropy"])
        try:
            # Fit the model
            nn_class_fit = nn_class.fit(X_train_scaled, y_train, validation_split = 0.2, epochs=500, batch_size=64, callbacks = call, verbose=0)
            b = nn_class_fit.history['val_accuracy'][-1]
            i = nn_class_fit.history['accuracy'][-1]
        except:
            # Fit the model
            nn_class_fit = nn_class.fit(X_train_scaled, y_train, validation_split = 0.2, epochs=500, batch_size=64, callbacks = call, verbose=0)
            b = nn_class_fit.history['val_accuracy'][-1]
            i = nn_class_fit.history['accuracy'][-1]

    # load a saved model
    saved_nn_class_model = load_model('best_nn_class_model.h5')
    
    return nn_class_fit, saved_nn_class_model