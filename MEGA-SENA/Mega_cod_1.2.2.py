import os

# Configuração do ambiente TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Dense, Input, Conv1D, MaxPooling1D, BatchNormalization, Dropout, Lambda, Flatten
)
from tensorflow.keras.optimizers import Adam

# Funções de Ajuste
def aplicar_ajuste(valor, ajuste):
    valor += ajuste
    while valor > 60:
        valor -= 60
    while valor < 1:
        valor += 60
    return valor

def calcular_ajuste_reverso(seq_real, seq_referencia):
    """
    Calcula os ajustes necessários para transformar a sequência real na sequência de referência.
    Aplica a regra de ajustes fixos considerando o intervalo de 1 a 60.
    """
    ajustes_necessarios = np.array(seq_real) - np.array(seq_referencia)

    # Aplica os ajustes de forma vetorizada
    ajustes_necessarios = np.where(ajustes_necessarios > 60, ajustes_necessarios - 60, ajustes_necessarios)
    ajustes_necessarios = np.where(ajustes_necessarios < 1, ajustes_necessarios + 60, ajustes_necessarios)

    return ajustes_necessarios


# Função para carregar dados do CSV
def carregar_dados(filepath):
    df = pd.read_csv(filepath, sep=';')
    df = df[['N1', 'N2', 'N3', 'N4', 'N5', 'N6']]
    return df.values

# Função para criar o modelo
def criar_modelo():
    modelo = Sequential([
        Input(shape=(6, 1)),  # Entrada ajustada para (6, 1)
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),  # Achata a saída para ser compatível com Dense
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(6, activation='linear'),  # Saída para prever 6 números
    ])
    modelo.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return modelo

# Função para calcular previsões
def calcular_previsao(seq_referencia, ajustes):
    entradas = []
    for ajuste in ajustes:
        previsao = [aplicar_ajuste(seq_referencia[i], ajuste[i]) for i in range(len(seq_referencia))]
        entradas.append(previsao)
    entradas = np.array(entradas)
    entradas = entradas.reshape((entradas.shape[0], entradas.shape[1], 1))  # Ajusta para (N, 6, 1)
    return entradas


# Treinamento supervisionado com cálculos automáticos
def treinamento_supervisionado_com_calculos_auto(modelo, sequencias, start_line=15, max_tentativas=3):
    for i in range(start_line, len(sequencias)):
        seq_referencia = sequencias[i - 1]
        seq_real = sequencias[i]
        ajustes = np.random.randint(-9, 10, size=(max_tentativas, len(seq_referencia)))

        entradas = calcular_previsao(seq_referencia, ajustes)
        entradas = np.expand_dims(entradas, axis=-1)  # Ajustar a forma para Conv1D

        previsoes = modelo.predict(entradas)
        for tentativa, previsao in enumerate(previsoes):
            previsao_ordenada = np.clip(np.round(previsao), 1, 60).astype(int)
            if np.array_equal(sorted(previsao_ordenada), sorted(seq_real)):
                print(f"Acertou: {previsao_ordenada}")
                break
        else:
            ajustes_necessarios = calcular_ajuste_reverso(seq_real, seq_referencia)
            print(f"Ajustes necessários: {ajustes_necessarios}")

# Função principal
if __name__ == "__main__":
    caminho_dados = "MEGA-SENA/dados_megasena/Mega_Sena.csv"
    sequencias = carregar_dados(caminho_dados)

    try:
        modelo = load_model("modelo_megasena.keras")
    except:
        modelo = criar_modelo()

    treinamento_supervisionado_com_calculos_auto(modelo, sequencias, start_line=15, max_tentativas=3)
    modelo.save("modelo_megasena.keras")
