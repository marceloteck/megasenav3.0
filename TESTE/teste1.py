print("Inicializando programa \n")
import os   
os.system('title Mega Sena Trainer v1.3')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
import random

seed_value = 42
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

import pandas as pd
import keras_tuner as kt
import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, Lambda, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

def calculate_differences(previous_draw, current_draw):
    """Calcula a diferença absoluta entre dois sorteios consecutivos."""
    return np.abs(np.array(current_draw) - np.array(previous_draw))

def build_model(input_shape):
    """Construi a rede neural para prever as diferenças ajustadas."""
    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=input_shape))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(6))  # Se for previsão de 6 números
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(X_train, y_train):
    """Treina a rede neural com as diferenças de sorteios."""
    model = build_model(X_train.shape[1:])
    model.fit(X_train, y_train, epochs=50, batch_size=32)
    return model

def adjust_predictions(predictions, corrections):
    """Ajusta as previsões com base nas correções aprendidas."""
    return predictions + corrections

def prepare_training_data(all_data):
    """Prepara os dados de treino a partir dos sorteios históricos."""
    X_train, y_train = [], []
    
    for i in range(len(all_data) - 1):
        previous_draw = all_data[i]
        current_draw = all_data[i + 1]
        
        # Calcula as diferenças entre o sorteio atual e o anterior
        diff = calculate_differences(previous_draw, current_draw)
        
        # Aqui você pode definir como a correção deve ser aplicada
        correction = np.random.choice([-1, 1, -3, 3, -6, 6, -9, 9])
        
        X_train.append(diff)
        y_train.append(correction)  # O modelo aprende a previsão de correção
    
    return np.array(X_train), np.array(y_train)

# Exemplo de uso:
# Dados de sorteios passados (cada sorteio tem 6 números)
all_data = [
    [15, 18, 27, 31, 39, 42],
    [10, 21, 32, 38, 51, 58],
    # Adicione mais dados conforme necessário
]

X_train, y_train = prepare_training_data(all_data)
model = train_model(X_train, y_train)

# Exemplo de previsão para o próximo sorteio
last_draw = all_data[-1]
last_diff = calculate_differences(all_data[-2], last_draw)
last_diff = last_diff.reshape(1, -1)

predicted_corrections = model.predict(last_diff)

# Ajuste a previsão de acordo com as correções
predicted_next_draw = last_draw + predicted_corrections
print("Próximo sorteio previsto:", predicted_next_draw)
