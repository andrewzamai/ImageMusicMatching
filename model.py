'''
Image-Music matching main

TODO: clean more matching pairs with not present songs IDs: DONE

1) Datasets loading: we will use the matching txt files to create our numpy matrices
    eg. lines i of train_matching_cleaned.txt will create sample X_train[i] where its obtained as concatenation of feature vector extracted from pretrained CNN (+else see Building emotional Machines paper) and music feature vector 
    target is third field of line
    
    store them in csv format so to not be every time computed and easily uploaded into numpy matrix

2) define model, early stopping, monitor loss curves (define function)

'''


from pyexpat.errors import XML_ERROR_INVALID_TOKEN
import pandas as pd
import numpy as np
import csv
import os

from sklearn import preprocessing

import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
from tensorflow import keras  # tf.keras

tf.random.set_seed(23)

from pickle import dump

def plot_learning_acc_and_loss(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.show()


def trainModel():
    pathToFeaturesCSVFolder = '/Users/andrew/Projects/ImageMusicMatching/IMEMNet_PairsTxt/FeaturesCSV'

    train_df = pd.read_csv(pathToFeaturesCSVFolder + '/trainCSV.csv', sep=',', header=None)
    train_np = pd.DataFrame(train_df).to_numpy()
    X_train, Y_train = train_np[:, :-1], train_np[:, -1]
    del train_df


    valid_df = pd.read_csv(pathToFeaturesCSVFolder + '/validCSV.csv', sep=',', header=None)
    valid_np = pd.DataFrame(valid_df).to_numpy()
    X_valid, Y_valid= valid_np[:, :-1], valid_np[:, -1]
    del valid_df


    test_df = pd.read_csv(pathToFeaturesCSVFolder + '/testCSV.csv', sep=',', header=None)
    test_np = pd.DataFrame(test_df).to_numpy()
    X_test, Y_test= test_np[:, :-1], test_np[:, -1]
    del test_df


    print(X_train)
    print(Y_train)

    print(X_valid)
    print(Y_valid)

    print(X_test)
    print(Y_test)


    # Standardizing Features
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train) #scaled training data
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    print(X_train_scaled)
    print(Y_train)

    print(X_valid_scaled)
    print(Y_valid)

    print(X_test_scaled)
    print(Y_test)


    # Build Model
    # Define metric
    # Define function to plot accuracy and test loss

    input_n_units = X_train_scaled.shape[1]

    model = keras.models.Sequential([
        tf.keras.Input(shape = (input_n_units,)),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    #model.summary()
    #keras.utils.plot_model(model, "my_model.png", show_shapes=True)

    model.compile(loss = tf.keras.losses.MeanSquaredError(),
                optimizer = keras.optimizers.Adam(learning_rate=1e-3),
                metrics=[tf.keras.losses.MeanSquaredError()])

    # Define model architecture and optimizer
    history = model.fit(X_train_scaled, Y_train, epochs=20,
                        validation_data=(X_valid_scaled, Y_valid))


    plot_learning_acc_and_loss(history)


    print(model.evaluate(X_test_scaled, Y_test))

    model.save('/Users/andrew/Projects/ImageMusicMatching/srcCode/ModelSrcCode/imageMusicTrainedModel.h5')
    dump(scaler, open('/Users/andrew/Projects/ImageMusicMatching/srcCode/ModelSrcCode/scaler.pkl', 'wb'))

    return(model, scaler)


trainModel()








