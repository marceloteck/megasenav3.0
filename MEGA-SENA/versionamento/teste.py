print("Inicializando programa \n")
import os   
os.system('title Mega Sena Trainer v1.1')
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
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.utils.multiclass import type_of_target



# Habilitar a desserialização insegura
keras.config.enable_unsafe_deserialization()

def build_model(hp):
    model = Sequential([
        Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
        Conv1D(filters=hp.Int('filters_1', min_value=32, max_value=256, step=32), kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        Conv1D(filters=hp.Int('filters_2', min_value=32, max_value=256, step=32), kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        LSTM(units=hp.Int('lstm_units', min_value=50, max_value=300, step=50), return_sequences=False),
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
        max_epochs=20,
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
    if os.path.exists(processed_files_filename):
        with open(processed_files_filename, 'r') as f:
            processed_files = set(f.read().splitlines())
    else:
        processed_files = set()

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

    if not data.empty and data.size > 0: 
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

def augment_data(X, y, n_augmentations=5):
    X_augmented = []
    y_augmented = []
    for _ in range(n_augmentations):
        noise = np.random.normal(0, 0.1, X.shape)
        X_augmented.append(X + noise)
        y_augmented.append(y)
    X_augmented = np.vstack(X_augmented)
    y_augmented = np.vstack(y_augmented)
    return X_augmented, y_augmented

def balance_data(X, y):
    target_type = type_of_target(y)  # Verificar tipo de alvo
    print(f"Tipo de alvo: {target_type}")

    # Apenas aplique SMOTE para classificação
    if target_type in ['binary', 'multiclass']:
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X.reshape(X.shape[0], -1), y)
        X_resampled = X_resampled.reshape(X_resampled.shape[0], X.shape[1], 1)
        return X_resampled, y_resampled
    else:
        print("SMOTE não é aplicável para tarefas de regressão.")
        return X, y


def avaliation_modell(X_test_scaled, y_test_scaled):
    y_pred_scaled = best_model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_rescaled = scaler_y.inverse_transform(y_test_scaled)

    mse = mean_squared_error(y_test_rescaled, y_pred)
    mae = mean_absolute_error(y_test_rescaled, y_pred)
    print(f"Erro Médio Quadrado (MSE): {mse}")
    print(f"Erro Absoluto Médio (MAE): {mae}")

folder_path = 'MEGA-SENA/dados_megasena/'
model_path = 'MEGA-SENA/megasena_model_automatic_training.2.0.keras'

X_filename='MEGA-SENA/prepared_data_X.npy'
y_filename='MEGA-SENA/prepared_data_y.npy'
processed_files_filename='MEGA-SENA/processed_files.txt'

all_data = load_and_preprocess_data(folder_path, processed_files_filename, X_filename)
X, y = prepare_data(all_data, X_filename, y_filename)

# Verificar o tipo de alvo
target_type = type_of_target(y)
print(f'Tipo de alvo: {target_type}')  # Verifique o tipo de `y`

if target_type in ['binary', 'multiclass']:
    # Só aplique SMOTE se for tarefa de classificação
    X, y = balance_data(X, y)
else:
    print("SMOTE não é aplicável para tarefas de regressão.")


X, y = augment_data(X, y)
X, y = balance_data(X, y)

dados = load_dados(folder_path)

print("Iniciando treinamento...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = MinMaxScaler(feature_range=(1, 60))
scaler_y = MinMaxScaler(feature_range=(1, 60))
X_train_scaled = scaler_X.fit_transform(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_scaled = scaler_X.transform(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, val_index in kf.split(X_train_scaled):
    X_train_k, X_val_k = X_train_scaled[train_index], X_train_scaled[val_index]
    y_train_k, y_val_k = y_train_scaled[train_index], y_train_scaled[val_index]
    
    tuner, best_hps = tune_hyperparameters(X_train_k, y_train_k)
    best_model = tuner.hypermodel.build(best_hps)
    history = best_model.fit(X_train_k, y_train_k, epochs=100, validation_data=(X_val_k, y_val_k), batch_size=16, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

tuner, best_hps = tune_hyperparameters(X_train_scaled, y_train_scaled)

if os.path.exists(model_path) and all_data.empty:
    print("\nCarregando modelo salvo...")
    best_model = load_model(model_path)
    
    avaliation_modell(X_test_scaled, y_test_scaled)    
else:
    print("\nTreinando o melhor modelo...")
    best_model = tuner.hypermodel.build(best_hps)
    history = best_model.fit(X_train_scaled, y_train_scaled, epochs=100, validation_split=0.2, batch_size=16, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

print("\nTreinamento concluído.")
avaliation_modell(X_test_scaled, y_test_scaled)

print("Salvando o modelo...")
best_model.save(model_path)
print("Modelo salvo com sucesso em:", model_path)
