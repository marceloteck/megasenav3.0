import os

# Configurações do ambiente
os.system('title Mega Sena Trainer v1.5.3')
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
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Lambda, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from collections import Counter



# Sementes para reproducibilidade
seed_value = 42
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

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
    """Carrega e processa os dados CSV na pasta fornecida"""
    all_data = pd.concat(
        (pd.read_csv(os.path.join(folder_path, file), sep=';')
         .assign(Data=lambda df: pd.to_datetime(df['Data'], format='%d/%m/%Y'))
         for file in os.listdir(folder_path) if file.endswith('.csv')),
        ignore_index=True
    ).sort_values('Data')
    return all_data

def load_and_preprocess_data(folder_path, processed_files_filename, X_filename):
    """Carrega e pré-processa os dados a partir dos arquivos CSV e os prepara para treinamento"""
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
    """Prepara os dados para treinamento, considerando os sorteios anteriores"""
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
        X_new = np.array([data.iloc[i - 12:i, 1:].values.flatten() for i in tqdm(range(12, len(data)), desc="Preparando dados", unit="iteração")])
        y_new = np.array([data.iloc[i, 1:].values for i in range(12, len(data))])

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

# Função para gerar dados sintéticos de acordo com a frequência dos números sorteados
def gerar_dados_sinteticos(dados_reais, num_amostras=1000):
    todos_numeros = dados_reais.iloc[:, 1:].values.flatten()
    todos_numeros = [num for num in todos_numeros if num != 0]
    frequencia = Counter(todos_numeros)
    total_numeros = sum(frequencia.values())
    
    probabilidade = {num: freq / total_numeros for num, freq in frequencia.items()}
    
    datas_existentes = set(dados_reais['Data'].dt.strftime('%d/%m/%Y'))
    
    dados_sinteticos = []
    for i in range(num_amostras):
        tentativas = 0
        limite_tentativas = 100  # Limite para evitar loop infinito
        
        while tentativas < limite_tentativas:
            numeros = random.choices(
                population=list(probabilidade.keys()), 
                weights=list(probabilidade.values()), 
                k=6
            )
            
            if 100 <= sum(numeros) <= 300:
                numeros.sort()  
                
                dia = random.randint(1, 28)
                mes = random.randint(1, 12)
                ano = random.randint(1996, 2024)
                data_ficticia = f"{dia:02d}/{mes:02d}/{ano}"
                
                if data_ficticia not in datas_existentes:
                    break
            tentativas += 1
        
        if tentativas < limite_tentativas:
            dados_sinteticos.append([data_ficticia] + numeros)
            datas_existentes.add(data_ficticia)
    
    df_sinteticos = pd.DataFrame(dados_sinteticos, columns=["Data"] + [f"Numero{i+1}" for i in range(6)])
    return df_sinteticos

# Função para construir o modelo
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

# Função para ajustar os hiperparâmetros
def tune_hyperparameters(X_train, y_train):
    tuner = kt.Hyperband(build_model, objective='val_loss', max_epochs=50, factor=3, directory='Memory', project_name='Memory_megasena')
    tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
    return tuner, tuner.get_best_hyperparameters(num_trials=1)[0]

# Função de avaliação do modelo
def evaluate_model(X_test_scaled, y_test_scaled, best_model):
    y_pred_scaled = best_model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_rescaled = scaler_y.inverse_transform(y_test_scaled)

    mse = mean_squared_error(y_test_rescaled, y_pred)
    mae = mean_absolute_error(y_test_rescaled, y_pred)
    print(f"Erro Médio Quadrado (MSE): {mse}")
    print(f"Erro Absoluto Médio (MAE): {mae}")
    
# Função para prever os próximos números da Mega-Sena
def predict_next_draw(model, dados_reais, scaler, scaler_y):
    """
    Gera uma previsão para o próximo sorteio com base nos dados mais recentes.
    """
    # Prepara os dados mais recentes para a entrada no modelo
    last_draw = dados_reais.iloc[-12:, 1:].values.flatten().reshape(1, 12, 6)
    last_draw_scaled = scaler.transform(last_draw.reshape(-1, 1)).reshape(1, 12, 6)

    # Faz a previsão
    predicted_scaled = model.predict(last_draw_scaled)
    predicted = scaler_y.inverse_transform(predicted_scaled)

    # Formata a previsão de forma legível
    predicted_numbers = predicted.flatten().astype(int)
    predicted_numbers = np.sort(predicted_numbers)
    
    return predicted_numbers

# Função para exibir os números sugeridos
def display_predicted_numbers(predicted_numbers):
    """
    Exibe os números previstos de forma legível.
    """
    print("\nNúmeros sugeridos para o próximo sorteio:")
    print(" | ".join(map(str, predicted_numbers)))    

# Execução principal
if __name__ == "__main__":
    print("Inicializando programa \n")

    all_data = load_and_preprocess_data(folder_path, processed_files_filename, X_filename)
    dados = load_dados(folder_path)

    X, y = prepare_data(dados, X_filename, y_filename, force_prepare=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_value)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    tuner, best_hyperparameters = tune_hyperparameters(X_train_scaled, y_train_scaled)
    best_model = tuner.hypermodel.build(best_hyperparameters)
    best_model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test_scaled))
    
    evaluate_model(X_test_scaled, y_test_scaled, best_model)
    
        # Prevendo os próximos números com base no modelo treinado
    predicted_numbers = predict_next_draw(best_model, dados, scaler, scaler_y)

    # Exibindo os números sugeridos
    display_predicted_numbers(predicted_numbers)
