import os

# Configuração do ambiente TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Dense, Input, Conv1D, MaxPooling1D, BatchNormalization, Dropout, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from tqdm import tqdm

# Função para criar o modelo
def criar_modelo():
    modelo = Sequential([
        Input(shape=(18, 1)),  # Entrada com formato (18, 1) para usar Conv1D
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(6, activation='linear')  # Saída para prever 6 números
    ])
    modelo.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return modelo

# Função para carregar dados do CSV
def carregar_dados(filepath):
    df = pd.read_csv(filepath, sep=';')
    df = df[['N1', 'N2', 'N3', 'N4', 'N5', 'N6']]
    return df.values

# Função para calcular previsões com base em ajustes
def calcular_previsao(seq_referencia, ajustes):
    entradas = []
    for ajuste in ajustes:
        previsao = seq_referencia + ajuste
        previsao = np.clip(previsao, 1, 60)  # Garantir que os números estejam no intervalo válido
        diferenca = previsao - seq_referencia
        somas = [sum(map(int, str(abs(num)))) for num in seq_referencia] + \
                [sum(map(int, str(abs(num)))) for num in previsao]
        entrada = np.concatenate((diferenca, somas))
        entrada = np.pad(entrada, (0, max(0, 18 - len(entrada))), 'constant')[:18]
        entradas.append(entrada)
    return np.array(entradas)

# Função para calcular ajustes necessários
def calcular_ajustes_necessarios(previsao, real):
    return np.array(real) - np.array(previsao)

# Treinamento supervisionado
def treinamento_supervisionado(modelo, sequencias, start_line=15, max_tentativas=3):
    acertos_por_quantidade = {i: 0 for i in range(7)}  # De 0 a 6 acertos

    for i in tqdm(range(start_line, len(sequencias)), desc="Treinamento supervisionado"):
        seq_referencia = sequencias[i - 1]
        seq_real = sequencias[i]
        ajustes = np.random.randint(-5, 6, size=(max_tentativas, len(seq_referencia)))
        entradas = calcular_previsao(seq_referencia, ajustes)
        entradas = np.expand_dims(entradas, axis=-1)  # Ajustar a forma para Conv1D
        previsoes = modelo.predict(entradas, verbose=0)

        for tentativa, previsao in enumerate(previsoes):
            previsao_ordenada = np.clip(np.round(previsao), 1, 60).astype(int)
            acertos = len(set(previsao_ordenada) & set(seq_real))
            acertos_por_quantidade[acertos] += 1

            if np.array_equal(sorted(previsao_ordenada), sorted(seq_real)):
                break
        else:
            previsao_final = np.clip(np.round(previsoes[-1]), 1, 60).astype(int)
            ajustes_necessarios = calcular_ajustes_necessarios(previsao_final, seq_real)
            entradas_treinamento = calcular_previsao(seq_referencia, [ajustes_necessarios])
            entradas_treinamento = np.expand_dims(entradas_treinamento, axis=-1)
            modelo.fit(entradas_treinamento, np.tile(seq_real, (1, 1)), epochs=1, verbose=0)

    print("\nRelatório de Acertos:")
    for quantidade, total in acertos_por_quantidade.items():
        print(f"Acertos {quantidade}: {total} vezes")

# Função principal
if __name__ == "__main__":
    # Caminho para o arquivo CSV
    caminho_dados = "MEGA-SENA/dados_megasena/Mega_Sena.csv"
    sequencias = carregar_dados(caminho_dados)

    # Carregar ou criar o modelo
    try:
        modelo = load_model("modelo_megasena.keras")
        print("Modelo carregado com sucesso!")
    except:
        modelo = criar_modelo()
        print("Modelo criado!")

    # Configurar Early Stopping
    early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

    # Treinamento supervisionado
    treinamento_supervisionado(modelo, sequencias, start_line=15, max_tentativas=3)

    # Salvar o modelo treinado
    #modelo.save("modelo_megasena.keras")
   # print("Modelo salvo com sucesso!")
