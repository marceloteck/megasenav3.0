# Imports e configurações iniciais
import os
# Configurações do ambiente
os.system('title LOTOFACIL Trainer v1.5')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import random
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (Dense, LSTM, Conv1D, MaxPooling1D, Dropout, 
                                     Lambda, Input, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations



# Sementes para reproducibilidade
seed_value = 42
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

# Habilitação da desserialização insegura
keras.config.enable_unsafe_deserialization()

# Variáveis globais e configurações
new_trainer = "noTrainer"  # Opções: "yesTrainer" | "noTrainer"
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_path = 'LOTOFACIL/dados_lotofacil/'
X_filename = 'LOTOFACIL/BaseDados/prepared_data_X.npy'
y_filename = 'LOTOFACIL/BaseDados/prepared_data_y.npy'
processed_files_filename = 'LOTOFACIL/processed_files.txt'
model_path = ('LOTOFACIL/Trainner/lotofacil_model_training.keras' if new_trainer == "noTrainer" 
              else f'LOTOFACIL/Trainner/lotofacil_model_training.{current_time}.keras')

# Funções utilitárias e de processamento de dados
def load_dados(folder_path):
    all_data = pd.concat(
        (pd.read_csv(os.path.join(folder_path, file), sep=';')
         .assign(Data=lambda df: pd.to_datetime(df['Data'], format='%d/%m/%Y'))
         for file in os.listdir(folder_path) if file.endswith('.csv')),
        ignore_index=True
    ).sort_values('Data')
    return all_data

def load_and_preprocess_data(folder_path, processed_files_filename, X_filename):
    processed_files = set()
    if os.path.exists(processed_files_filename):
        with open(processed_files_filename, 'r') as f:
            processed_files = set(f.read().splitlines())

    new_files = [file for file in os.listdir(folder_path) 
                 if file.endswith('.csv') and file not in processed_files or not os.path.exists(X_filename)]
    all_data = [pd.read_csv(os.path.join(folder_path, file), sep=';').assign(Data=lambda df: pd.to_datetime(df['Data'], format='%d/%m/%Y')) 
                for file in new_files]

    if all_data:
        all_data = pd.concat(all_data, ignore_index=True).sort_values('Data')

    with open(processed_files_filename, 'a') as f:
        for file in new_files:
            f.write(file + '\n')

    return all_data

def prepare_data(data, X_filename, y_filename, force_prepare=False):
    if isinstance(data, list):
        data = pd.DataFrame(data)
        
    if not force_prepare and os.path.exists(X_filename) and os.path.exists(y_filename):
        print("Carregando dados preparados dos arquivos...\n")
        X_existing = np.load(X_filename, allow_pickle=True)
        y_existing = np.load(y_filename, allow_pickle=True)
    else:
        print("Preparando os dados...")
        X_existing, y_existing = np.array([]), np.array([])

    if not data.empty:
        print("Adicionando novos dados...")
        diasAnalise = 40
        X_new = np.array([data.iloc[i - diasAnalise:i, 1:].values.flatten() for i in tqdm(range(diasAnalise, len(data)), desc="Preparando dados", unit="iteração")])
        y_new = np.array([data.iloc[i, 1:].values for i in range(diasAnalise, len(data))])

        if X_existing.size > 0:
            X = np.concatenate((X_existing, X_new))
            y = np.concatenate((y_existing, y_new))
        else:
            X, y = X_new, y_new

        print("Salvando os dados preparados em arquivos...")
        np.save(X_filename, X, allow_pickle=True)
        np.save(y_filename, y, allow_pickle=True)
    else:
        print("Nenhum dado novo para adicionar.")
        X, y = X_existing, y_existing

    return X, y

# Funções de treinamento e avaliação
def build_model(hp):
    model = Sequential([
        Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
        Conv1D(filters=hp.Int('filters_1', 32, 256, 32), kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        Conv1D(filters=hp.Int('filters_2', 32, 256, 32), kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        LSTM(units=hp.Int('lstm_units', 50, 400, 50), return_sequences=False),
        Dropout(hp.Float('dropout', 0.1, 0.5, 0.1)),
        Dense(units=hp.Int('dense_units', 32, 128, 32), activation='relu'),
        Dense(15),
        Lambda(lambda x: 1 + 24 * x)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                  loss='mean_squared_error')
    return model

def tune_hyperparameters(X_train, y_train):
    tuner = kt.Hyperband(build_model, objective='val_loss', max_epochs=50, factor=3, directory='Memory', project_name='Memory_lotofacil')
    tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
    return tuner, tuner.get_best_hyperparameters(num_trials=1)[0]

def evaluate_model(X_test_scaled, y_test_scaled, best_model):
    y_pred_scaled = best_model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_rescaled = scaler_y.inverse_transform(y_test_scaled)

    mse = mean_squared_error(y_test_rescaled, y_pred)
    mae = mean_absolute_error(y_test_rescaled, y_pred)
    print(f"Erro Médio Quadrado (MSE): {mse}")
    print(f"Erro Absoluto Médio (MAE): {mae}")
    
    escrever = f"\n\nMAIS UM RESULTADO:\nErro Médio Quadrado (MSE): {mse}\nErro Absoluto Médio (MAE): {mae}"
    with open("LOTOFACIL/Resultados_lotofacil.txt", 'a') as f:
        f.write(escrever)


# Execução principal
if __name__ == "__main__":
    print("Inicializando programa \n")

    all_data = load_and_preprocess_data(folder_path, processed_files_filename, X_filename)
    dados = load_dados(folder_path)

    X, y = prepare_data(all_data, X_filename, y_filename)
    

    print("Iniciando treinamento...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = MinMaxScaler(feature_range=(1, 25))
    scaler_y = MinMaxScaler(feature_range=(1, 25))
    X_train_scaled = scaler_X.fit_transform(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_scaled = scaler_X.transform(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    if os.path.exists(model_path):
        print("\nCarregando modelo salvo...")
        best_model = load_model(model_path)
    else:
        print("\nTreinando o melhor modelo...")
        tuner, best_hps = tune_hyperparameters(X_train_scaled, y_train_scaled)
        best_model = tuner.hypermodel.build(best_hps)
        best_model.fit(X_train_scaled, y_train_scaled, epochs=500, validation_split=0.2, batch_size=16, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

    ErrMse_trainer = 100
    Tentativa = 1

    while ErrMse_trainer > 50:
        print(f"\nTENTATIVA: {Tentativa}\nContinuando o treinamento com os novos dados...")
        best_model.fit(X_train_scaled, y_train_scaled, epochs=200, validation_split=0.2, batch_size=16, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

        evaluate_model(X_test_scaled, y_test_scaled, best_model)
        best_model.save(model_path)
        print("Modelo salvo.")

        print("\nPrevisão dos próximos números...")
        diasAnalise = 40
        last_five_draws = dados.iloc[-diasAnalise:, 1:].values.flatten()
        last_five_scaled = scaler_X.transform(last_five_draws.reshape(1, -1)).reshape(1, diasAnalise * 15, 1)
        predicted_scaled = best_model.predict(last_five_scaled)
        predicted_numbers = np.clip(scaler_y.inverse_transform(predicted_scaled).astype(int), 1, 25)
        print("\nNúmeros previstos para o próximo sorteio:", predicted_numbers[0])
        print("\nNúmeros previstos para o próximo sorteio 2:", predicted_scaled[0].astype(int))
        
        escrever = f"\nNúmeros previstos para o próximo sorteio: {predicted_numbers[0]}\nTambém: {predicted_scaled[0].astype(int)}"
        with open("LOTOFACIL/Resultados_lotofacil.txt", 'a') as f:
            f.write(escrever)

        if ErrMse_trainer <= 50:
            break
        if Tentativa >= 100:
            break
        Tentativa += 1
 



  