import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sl
import scipy as sc
import math as ma
from scipy import linalg, optimize, constants, interpolate, special, stats
from math import exp, pow, sqrt, log
import os

# only use GPU 3
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import seaborn as sns  # specialized graphical representations
import statsmodels.api as sm
import statsmodels.stats.api as sms

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn import neighbors
from sklearn.model_selection import cross_val_score

import tensorflow as tf
from itertools import product

os.environ["TF_NUM_INTRAOP_THREADS"] = "128"  # Threads within a single operation
os.environ["TF_NUM_INTEROP_THREADS"] = "128"  # Threads for parallel operations between different operations
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam, RMSprop  # Optimizers
from tensorflow.keras.regularizers import L1, L2, L1L2
from tensorflow.keras.utils import to_categorical, plot_model

Daten = pd.read_csv('./Flut-Daten.csv')
# Create target variables for the next 1-6 hours
for i in range(1, 7):
    Daten[f"Q_target_{i}"] = Daten["Q"].shift(-i)

# Remove missing values (e.g., the last 6 rows)
Daten = Daten.dropna()
Daten.columns

## here the data still needs to be split into training and test sets
X = Daten.drop(columns=["Q"] + [f"Q_target_{i}" for i in range(1, 7)])
y = Daten[[f"Q_target_{i}" for i in range(1, 7)]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Prepare data and parameters
output_steps = 6  # Number of prediction steps (1-6 hours)
feature_count = X_train.shape[1]

# Define parameter combinations
dropout_rates = [0.0, 0.1, 0.2, 0.3]
hidden_layers = [2, 3, 4]
neurons_per_layer = [64, 128]
activation_functions = ['relu', 'tanh', 'elu']
learning_rates = [0.001, 0.01]
batch_sizes = [2048]
regularizers = [L2(l2=0.01), L1L2(l1=0.01, l2=0.01)]
optimizers = ['Adam', 'SGD']
# losses = ['mean_squared_error']

# Generate all combinations
param_combinations = list(product(
    dropout_rates, hidden_layers, neurons_per_layer, activation_functions,
    learning_rates, batch_sizes, regularizers, optimizers
))

# Store results
results = []
# Training and evaluation
for idx, (dropout, layers, neurons, activation, lr, batch_size, reg, opt) in enumerate(
    product(dropout_rates, hidden_layers, neurons_per_layer, activation_functions, learning_rates, batch_sizes, regularizers, optimizers)
):
    print(f"Training Model {idx+1}/{len(param_combinations)}: Dropout={dropout}, Layers={layers}, Neurons={neurons}, Activation={activation}, LR={lr}, Batch={batch_size}, Regularizer={reg}, Optimizer={opt}")
    
    # Select optimizer
    if opt == 'Adam':
        optimizer = Adam(learning_rate=lr)
    elif opt == 'RMSprop':
        optimizer = RMSprop(learning_rate=lr)
    elif opt == 'SGD':
        optimizer = SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    
    # Create model
    model = Sequential()
    model.add(InputLayer(input_shape=(feature_count,)))
    for _ in range(layers):
        model.add(Dense(neurons, activation=None, kernel_regularizer=reg))
        model.add(BatchNormalization())  # Add Batch Normalization
        model.add(Activation(activation))  # Apply activation function
        model.add(Dropout(dropout))
    model.add(Dense(output_steps, activation='linear'))
    
    # Compile and train model
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
    
    # Learning rate schedule
    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=500,
        batch_size=batch_size,
        verbose=0,
        validation_data=(X_test, y_test),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            lr_schedule
        ]
    )
    
    # Calculate predictions and metrics
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # RMSE for the 1-6 hour prediction
    rmse_steps = [np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i])) for i in range(output_steps)]
    
    # Store results
    results.append({
        'Model': idx + 1,
        'Dropout': dropout,
        'Hidden Layers': layers,
        'Neurons per Layer': neurons,
        'Activation': activation,
        'Learning Rate': lr,
        'Batch Size': batch_size,
        'Regularizer': reg,
        'Optimizer': opt,
        'Loss Function': 'mean_squared_error',
        'R2': r2,
        'RMSE': rmse,
        'Train Loss': history.history['loss'][-1],
        'Val Loss': history.history['val_loss'][-1],
        **{f'RMSE Step {i+1}': rmse_steps[i] for i in range(output_steps)}
    })

# Save and display results in DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv('model_results.csv', index=False)
print("Model training complete. Results saved to 'model_results.csv'")