print("Inicializando programa \n")
import os   
os.system('title Mega Sena Trainer v1.4')
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

# Obter a data e hora atuais
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# Habilitar a desserialização insegura
keras.config.enable_unsafe_deserialization()

newTrainer = "noTrainner" # yesTrainner | noTrainner

def build_model(hp):
    model = Sequential([
        Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
        Conv1D(filters=hp.Int('filters_1', min_value=32, max_value=256, step=32), kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        Conv1D(filters=hp.Int('filters_2', min_value=32, max_value=256, step=32), kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        LSTM(units=hp.Int('lstm_units', min_value=50, max_value=400, step=50), return_sequences=False),
        Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)),
        Dense(units=hp.Int('dense_units', min_value=32, max_value=128, step=32), activation='relu'),
        Dense(6),
        Lambda(lambda x: 1 + 59 * x)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='mean_squared_error')
    return model

def tune_hyperparameters(X_train, y_train):
    tuner = kt.Hyperband(
        build_model,
        objective='val_loss',
        max_epochs=50,
        factor=3,
        directory='Memory',
        project_name='Memory_megasena'
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stopping])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return tuner, best_hps

def load_dados(folder_path):
    all_data = pd.concat(
        (pd.read_csv(os.path.join(folder_path, file), sep=';').assign(Data=lambda df: pd.to_datetime(df['Data'], format='%d/%m/%Y'))
         for file in os.listdir(folder_path) if file.endswith('.csv')),
        ignore_index=True
    ).sort_values('Data')
    return all_data

def load_and_preprocess_data(folder_path, processed_files_filename, X_filename):
    # Carregar a lista de arquivos já processados
    if os.path.exists(processed_files_filename):
        with open(processed_files_filename, 'r') as f:
            processed_files = set(f.read().splitlines())
    else:
        processed_files = set()

    # Identificar novos arquivos
    all_data = []
    new_files = []

    for file in os.listdir(folder_path):
        if file.endswith('.csv') and file not in processed_files or not os.path.exists(X_filename):
            file_path = os.path.join(folder_path, file)
            new_files.append(file)
            data = pd.read_csv(file_path, sep=';')
            data['Data'] = pd.to_datetime(data['Data'], format='%d/%m/%Y')
            all_data.append(data)

    if all_data:
        all_data = pd.concat(all_data, ignore_index=True).sort_values('Data')
    else:
        all_data = pd.DataFrame()

    # Atualizar o arquivo de arquivos processados
    with open(processed_files_filename, 'a') as f:
        for file in new_files:
            f.write(file + '\n')

    return all_data


def prepare_data(data, X_filename, y_filename, force_prepare=False):
    if not force_prepare and os.path.exists(X_filename) and os.path.exists(y_filename):
        print("Carregando dados preparados dos arquivos...\n") 
        X_existing = np.load(X_filename, allow_pickle=True)
        y_existing = np.load(y_filename, allow_pickle=True)
        
    else:
        print("Preparando os dados...")
        X_existing, y_existing = np.array([]), np.array([])

    if not data.empty and data.size > 0: # se data nao e vazio
        print("Adicionando novos dados...")
        X_new, y_new = [], []
        for i in tqdm(range(12, len(data)), desc="Preparando dados", unit="iteração"):
            X_new.append(data.iloc[i - 12:i, 1:].values.flatten()) 
            y_new.append(data.iloc[i, 1:].values)
        X_new, y_new = np.array(X_new), np.array(y_new)
        
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

def avaliation_modell(X_test_scaled, y_test_scaled):
    # Avaliação do modelo e salvamento
    y_pred_scaled = best_model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_rescaled = scaler_y.inverse_transform(y_test_scaled)

    mse = mean_squared_error(y_test_rescaled, y_pred)
    mae = mean_absolute_error(y_test_rescaled, y_pred)
    print(f"Erro Médio Quadrado (MSE): {mse}")
    print(f"Erro Absoluto Médio (MAE): {mae}")
    
###############################################

if newTrainer == "noTrainner":
    model_path = 'MEGA-SENA/Trainner/megasena_model_training.keras'
else:
    model_path = f'MEGA-SENA/Trainner/megasena_model_training.{current_time}.keras'

folder_path = 'MEGA-SENA/dados_megasena/'
X_filename = 'MEGA-SENA/BaseDados/prepared_data_X.npy'
y_filename = 'MEGA-SENA/BaseDados/prepared_data_y.npy'
processed_files_filename = 'MEGA-SENA/processed_files.txt'

all_data = load_and_preprocess_data(folder_path, processed_files_filename, X_filename)
X, y = prepare_data(all_data, X_filename, y_filename)

dados = load_dados(folder_path)

print("Iniciando treinamento...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = MinMaxScaler(feature_range=(1, 60))
scaler_y = MinMaxScaler(feature_range=(1, 60))
X_train_scaled = scaler_X.fit_transform(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_scaled = scaler_X.transform(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Carregar ou treinar o modelo
if os.path.exists(model_path):
    print("\nCarregando modelo salvo...")
    best_model = load_model(model_path)
else:
    print("\nTreinando o melhor modelo...")
    tuner, best_hps = tune_hyperparameters(X_train_scaled, y_train_scaled)
    best_model = tuner.hypermodel.build(best_hps)
    best_model.fit(
        X_train_scaled, 
        y_train_scaled, 
        epochs=500, 
        validation_split=0.2, 
        batch_size=16, 
        callbacks=[EarlyStopping(monitor='val_loss', patience=10)]
    )

# Continuar o treinamento com os novos dados
print("\nContinuando o treinamento com os novos dados...")
best_model.fit(
    X_train_scaled, 
    y_train_scaled, 
    epochs=100,  # Pode ajustar o número de épocas para evitar overfitting
    validation_split=0.2, 
    batch_size=16, 
    callbacks=[EarlyStopping(monitor='val_loss', patience=10)]
)

avaliation_modell(X_test_scaled, y_test_scaled)
best_model.save(model_path)
print("Modelo salvo.")

######################### Obter os últimos cinco sorteios -12
print("\n")
last_five_draws = dados.iloc[-12:, 1:].values.flatten()

# Normalizando e ajustando a forma para (1, 72, 1) - 12*6=72
last_five_scaled = scaler_X.transform(last_five_draws.reshape(1, -1))
last_five_scaled = last_five_scaled.reshape(1, 72, 1)

# Predição dos próximos números
predicted_scaled = best_model.predict(last_five_scaled)

# Desescalonando as previsões e arredondando para os números mais próximos
predicted_numbers = scaler_y.inverse_transform(predicted_scaled)
predicted_numbers = predicted_numbers.astype(int)

# Garantir que os números estão no intervalo de 1 a 60
predicted_numbers = np.clip(predicted_numbers, 1, 60)

print("\nNúmeros previstos para o próximo sorteio:", predicted_numbers[0])



###############################################################################
# Previsão dos próximos números
last_fifteen_draws = dados.iloc[-12:, 1:].values

# Normalizando e ajustando a forma para (1, 255)
last_fifteen_draws = last_fifteen_draws.reshape(1, -1)
last_five_scaled = last_five_scaled.reshape(1, 72, 1)

# Predição de várias amostras e coleta dos números mais frequentes
n_predictions = 1000
all_predicted_numbers = []

#for _ in range(n_predictions):
# Predição dos próximos números apenas uma vez
predicted_scaled = best_model.predict(last_five_scaled)
predicted_numbers = scaler_y.inverse_transform(predicted_scaled)
predicted_numbers = np.round(predicted_numbers).astype(int)
predicted_numbers = np.clip(predicted_numbers, 1, 60)

# Contar as frequências dos números previstos
number_counts = pd.Series(all_predicted_numbers).value_counts()
most_frequent_numbers = number_counts.head(30).index.tolist()  # Selecionar os 30 números mais frequentes

# Gerar 7 jogadas distintas a partir dos números mais frequentes
from itertools import combinations
from collections import Counter

# Contar a frequência de cada número no conjunto de treinamento
number_counts = Counter(y_train.flatten())

# Selecionar os 30 números mais frequentes
top_30_numbers = [num for num, _ in number_counts.most_common(50)]

# Formar todas as combinações possíveis de 15 números a partir dos 30 mais frequentes
all_combinations = list(combinations(top_30_numbers, 6))

# Garantir que há pelo menos 5 combinações para escolher
if len(all_combinations) < 10:
    jogadas = all_combinations  # Seleciona todas as combinações disponíveis
else:
    # Selecionar 8 combinações aleatórias
    eight_combinations = np.random.choice(range(len(all_combinations)), 10, replace=False)
    jogadas = [all_combinations[i] for i in eight_combinations]
    
    print("\n")
# Exibir as jogadas formadas
for i, jogada in enumerate(jogadas, start=1):
    jogada_int = [int(num) for num in jogada]  # Converter np.int64 para int
    print(f"Jogada {i}: {sorted(jogada_int)}")

###############################################################################




##################################### PLOTAGEM

# Comparação visual entre os números previstos e os números reais
def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    for i in range(6):  # Mega-Sena tem 6 números por sorteio
        plt.subplot(3, 2, i + 1)
        plt.plot(y_true[:, i], label='Real', color='blue')
        plt.plot(y_pred[:, i], label='Previsto', color='orange')
        plt.title(f'Número {i+1}')
        plt.xlabel('Sorteio')
        plt.ylabel('Valor')
        plt.legend()
    plt.tight_layout()
    plt.show()

"""
# Chamar a função com os dados reais e previstos
plot_predictions(y_test_rescaled, y_pred)


# Contagem da frequência de números reais sorteados no conjunto de treinamento
real_number_counts = Counter(y_train.flatten())

# Contagem da frequência de números previstos no conjunto de teste
predicted_number_counts = Counter(y_pred.flatten())

# Organizar as contagens para comparação
real_numbers = [num for num, _ in real_number_counts.most_common()]
real_frequencies = [real_number_counts[num] for num in real_numbers]
predicted_frequencies = [predicted_number_counts.get(num, 0) for num in real_numbers]

# Plotar as frequências dos números reais e previstos
plt.figure(figsize=(12, 6))
plt.bar(real_numbers, real_frequencies, alpha=0.7, label='Real', color='blue')
plt.bar(real_numbers, predicted_frequencies, alpha=0.7, label='Previsto', color='orange')
plt.xlabel('Número')
plt.ylabel('Frequência')
plt.title('Frequência dos Números: Real vs. Previsto')
plt.legend()
plt.show()
"""