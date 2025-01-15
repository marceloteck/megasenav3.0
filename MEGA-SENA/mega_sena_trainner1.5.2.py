# Imports e configurações iniciais
import os
# Configurações do ambiente
os.system('title Mega Sena Trainer v1.5.2')
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
folder_path = 'MEGA-SENA/dados_megasena/'
X_filename = 'MEGA-SENA/BaseDados/prepared_data_X.npy'
y_filename = 'MEGA-SENA/BaseDados/prepared_data_y.npy'
processed_files_filename = 'MEGA-SENA/processed_files.txt'
model_path = ('MEGA-SENA/Trainner/megasena_model_training.keras' if new_trainer == "noTrainer" 
              else f'MEGA-SENA/Trainner/megasena_model_training.{current_time}.keras')

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
        NTime = 36 # dias de analise
        X_new = np.array([data.iloc[i - NTime:i, 1:].values.flatten() for i in tqdm(range(NTime, len(data)), desc="Preparando dados", unit="iteração")])
        y_new = np.array([data.iloc[i, 1:].values for i in range(NTime, len(data))])

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
        Dense(6),
        Lambda(lambda x: 1 + 59 * x)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                  loss='mean_squared_error')
    return model

def tune_hyperparameters(X_train, y_train):
    tuner = kt.Hyperband(build_model, objective='val_loss', max_epochs=50, factor=3, directory='Memory', project_name='Memory_megasena')
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
    
def gerar_dados_sinteticos(dados_reais, num_amostras=1000):
    # Contar a frequência de cada número nos dados reais, excluindo o zero
    todos_numeros = dados_reais.iloc[:, 1:].values.flatten()
    todos_numeros = [num for num in todos_numeros if num != 0]
    frequencia = Counter(todos_numeros)
    total_numeros = sum(frequencia.values())
    
    # Converter as frequências em probabilidades
    probabilidade = {num: freq / total_numeros for num, freq in frequencia.items()}
    
    # Gerar dados sintéticos
    dados_sinteticos = []
    for i in range(num_amostras):
        tentativas = 0
        limite_tentativas = 100  # Define um limite para evitar loop infinito
        
        while tentativas < limite_tentativas:
            # Selecionar 6 números baseados nas probabilidades
            numeros = random.choices(
                population=list(probabilidade.keys()), 
                weights=list(probabilidade.values()), 
                k=6
            )
            
            # Verificar se os números estão dentro de uma soma realista
            if 100 <= sum(numeros) <= 300:
                numeros.sort()  # Ordenar os números em ordem crescente
                break
            tentativas += 1
        
        if tentativas < limite_tentativas:
            # Adicionar a data fictícia e a sequência ao conjunto sintético
            data_ficticia = f"{(i % 28) + 1:02d}/{(i % 12) + 1:02d}/2025"  # Gera datas fictícias no formato d/m/ano
            dados_sinteticos.append([data_ficticia] + numeros)
    
    # Converter para DataFrame para facilitar a manipulação posterior
    df_sinteticos = pd.DataFrame(dados_sinteticos, columns=["Data"] + [f"Numero{i+1}" for i in range(6)])
    df_sinteticos = df_sinteticos.reset_index(drop=True) 
    #df_sinteticos = df_sinteticos.to_string(index=False)
    
    return df_sinteticos


def add_features(data, NTime=36):
    data['media_movel'] = data.iloc[:, 1:].rolling(window=NTime).mean().mean(axis=1)
    data['desvio_padrao'] = data.iloc[:, 1:].rolling(window=NTime).std().mean(axis=1)
    for num in range(1, 61):
        data[f'freq_num_{num}'] = data.iloc[:, 1:].apply(lambda row: (row == num).sum(), axis=1)
    return data

def normalize_individual_draws(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data.iloc[:, 1:] = scaler.fit_transform(data.iloc[:, 1:])
    return data

def remove_outliers(data, min_sum=100, max_sum=300):
    data = data[(data.iloc[:, 1:].sum(axis=1) >= min_sum) & (data.iloc[:, 1:].sum(axis=1) <= max_sum)]
    return data

def add_year_feature(data):
    # Garantir que a coluna 'Data' seja convertida para datetime, caso não tenha sido feita antes
    data['Data'] = pd.to_datetime(data['Data'], format='%d/%m/%Y')
    
    # Adicionar a coluna 'ano' com o ano extraído da coluna 'Data'
    data['ano'] = data['Data'].dt.year
    
    return data


    

# Execução principal
if __name__ == "__main__":
    print("Inicializando programa \n")

    all_data = load_and_preprocess_data(folder_path, processed_files_filename, X_filename)
    dados = load_dados(folder_path)
    dados_sinteticos = gerar_dados_sinteticos(dados)

    if isinstance(all_data, list) and all_data:
        all_data = pd.concat(all_data, ignore_index=True)
    elif not all_data:
        all_data = pd.DataFrame()

    if all_data.empty:
        combined_data = dados_sinteticos
    else:
        combined_data = pd.concat([all_data, dados_sinteticos], ignore_index=True)

    # Adicionar a coluna 'ano'
    dados = add_year_feature(dados)

    # Verifique a saída para garantir que a coluna 'ano' foi adicionada corretamente
    print(dados.head())   
         

    # Adicione novas features
    combined_data = add_features(combined_data)
    combined_data = normalize_individual_draws(combined_data)
    combined_data = remove_outliers(combined_data)
    combined_data = add_year_feature(combined_data)

    combined_data_list = combined_data.values.tolist()
    X, y = prepare_data(combined_data_list, X_filename, y_filename, N=36)
    
    



    


    print("Iniciando treinamento...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = MinMaxScaler(feature_range=(1, 60))
    scaler_y = MinMaxScaler(feature_range=(1, 60))
    X_train_scaled = scaler_X.fit_transform(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_scaled = scaler_X.transform(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    if os.path.exists(model_path):
        print("\nCarregando modelo salvo...")
        best_model = load_model(model_path)
    #else:
    print("\nTreinando o melhor modelo...")
    tuner, best_hps = tune_hyperparameters(X_train_scaled, y_train_scaled)
    best_model = tuner.hypermodel.build(best_hps)
    best_model.fit(X_train_scaled, y_train_scaled, epochs=500, validation_split=0.2, batch_size=16, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

   # print("\nContinuando o treinamento com os novos dados...")
    #best_model.fit(X_train_scaled, y_train_scaled, epochs=100, validation_split=0.2, batch_size=16, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

    evaluate_model(X_test_scaled, y_test_scaled, best_model)
    best_model.save(model_path)
    print("Modelo salvo.")

    print("\nPrevisão dos próximos números...")
    last_five_draws = dados.iloc[-36:, 1:].values.flatten()
    last_five_scaled = scaler_X.transform(last_five_draws.reshape(1, -1)).reshape(1, 216, 1)
    predicted_scaled = best_model.predict(last_five_scaled)
    predicted_numbers = np.clip(scaler_y.inverse_transform(predicted_scaled).astype(int), 1, 60)
    print("\nNúmeros previstos para o próximo sorteio:", predicted_numbers[0])

    # Continuação do código de previsão...
 