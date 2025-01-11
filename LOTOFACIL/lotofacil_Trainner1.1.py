import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.system('title Lotofácil Trainer v1.1')

import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, Lambda, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import keras_tuner as kt

def build_model(hp):
    model = Sequential([
        Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
        Conv1D(filters=hp.Int('filters_1', min_value=32, max_value=256, step=32), kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=hp.Int('filters_2', min_value=32, max_value=256, step=32), kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(units=hp.Int('lstm_units', min_value=50, max_value=300, step=50), return_sequences=False),
        Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)),
        Dense(units=hp.Int('dense_units', min_value=32, max_value=128, step=32), activation='relu'),
        Dense(15),
        Lambda(lambda x: 1 + 24 * x)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='mean_squared_error')
    return model

def tune_hyperparameters(X_train, y_train):
    tuner = kt.Hyperband(
        build_model,
        objective='val_loss',
        max_epochs=20,
        factor=3,
        directory='Memory',
        project_name='Memory_lotofacil'
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stopping])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return tuner, best_hps

def load_and_preprocess_data(folder_path):
    all_data = pd.concat(
        (pd.read_csv(os.path.join(folder_path, file), sep=';').assign(Data=lambda df: pd.to_datetime(df['Data'], format='%d/%m/%Y'))
         for file in os.listdir(folder_path) if file.endswith('.csv')),
        ignore_index=True
    ).sort_values('Data')
    return all_data

def prepare_data(data, X_filename='LOTOFACIL/prepared_data_X.npy', y_filename='LOTOFACIL/prepared_data_y.npy', force_prepare=False):
    if not force_prepare and os.path.exists(X_filename) and os.path.exists(y_filename):
        print("Carregando dados preparados dos arquivos...")
        X = np.load(X_filename, allow_pickle=True)
        y = np.load(y_filename, allow_pickle=True)
    else:
        print("Preparando os dados...")
        X, y = [], []
        for i in tqdm(range(5, len(data)), desc="Preparando dados", unit="iteração"):
            X.append(data.iloc[i - 5:i, 1:].values.flatten())
            y.append(data.iloc[i, 1:].values)
        X, y = np.array(X), np.array(y)
        
        print("Salvando os dados preparados em arquivos...")
        np.save(X_filename, X, allow_pickle=True)
        np.save(y_filename, y, allow_pickle=True)
    
    return X, y

folder_path = 'LOTOFACIL/dados_lotofacil/'
model_path = 'LOTOFACIL/lotofacil_model_automatic_training.2.0.keras'

print("Processando informações...")
all_data = load_and_preprocess_data(folder_path)

print("Preparando dados...")
X, y = prepare_data(all_data)

print("Iniciando treinamento...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = MinMaxScaler(feature_range=(1, 25))
scaler_y = MinMaxScaler(feature_range=(1, 25))
X_train_scaled = scaler_X.fit_transform(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_scaled = scaler_X.transform(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

print("Busca de hiperparâmetros...")
tuner, best_hps = tune_hyperparameters(X_train_scaled, y_train_scaled)

print("Treinando o melhor modelo...")
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(X_train_scaled, y_train_scaled, epochs=3, validation_split=0.2, batch_size=16, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

# Avaliação do modelo e salvamento
y_pred_scaled = best_model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_rescaled = scaler_y.inverse_transform(y_test_scaled)

mse = mean_squared_error(y_test_rescaled, y_pred)
mae = mean_absolute_error(y_test_rescaled, y_pred)
print(f"Erro Médio Quadrado (MSE): {mse}")
print(f"Erro Absoluto Médio (MAE): {mae}")

best_model.save(model_path)
print("Modelo salvo.")

# Obter os últimos cinco sorteios
last_five_draws = all_data.iloc[-5:, 1:].values.flatten()

last_five_scaled = scaler_X.transform(last_five_draws.reshape(1, -1))
last_five_scaled = last_five_scaled.reshape(1, 75, 1)

# Predição dos próximos números
predicted_scaled = best_model.predict(last_five_scaled)

# Desescalonando as previsões e arredondando para os números mais próximos
predicted_numbers = scaler_y.inverse_transform(predicted_scaled)
predicted_numbers = np.round(predicted_numbers).astype(int)

# Garantir que os números estão no intervalo de 1 a 25
predicted_numbers = np.clip(predicted_numbers, 1, 25)

print("\nNúmeros previstos para o próximo sorteio:", predicted_numbers[0])
