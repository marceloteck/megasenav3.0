import os
# Configurações do ambiente
os.system('title Mega Código 1.0')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np



# Função para criar o modelo de rede neural
def criar_modelo():
    modelo = Sequential([
        Input(shape=(18,)),  # 6 diferenças + 6 somas + 6 somas da outra sequência
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(6)  # Saída para os 6 ajustes numéricos
    ])
    modelo.compile(optimizer='adam', loss='mse')
    return modelo

# Função para treinar o modelo
def treinar_modelo(modelo, sequencias_atuais, sequencias_anteriores, proximas_sequencias):
    entradas = []
    saidas = []
    for seq_atual, seq_ant, seq_prox in zip(sequencias_atuais, sequencias_anteriores, proximas_sequencias):
        diferenca = np.array(seq_ant) - np.array(seq_atual)
        somas = [sum(map(int, str(abs(num)))) for num in seq_ant] + \
                [sum(map(int, str(abs(num)))) for num in seq_atual]
        entradas.append(np.concatenate((diferenca, somas)))
        saidas.append(np.array(seq_prox) - np.array(seq_ant))  # Ajuste esperado para todos os números

    entradas = np.array(entradas)
    saidas = np.array(saidas)
    modelo.fit(entradas, saidas, epochs=500, verbose=0)  # Treinamento com 500 épocas

# Função para prever os próximos números
def prever_proximos_numeros(modelo, sequencia_atual, sequencia_anterior):
    diferenca = np.array(sequencia_anterior) - np.array(sequencia_atual)
    somas = [sum(map(int, str(abs(num)))) for num in sequencia_anterior] + \
            [sum(map(int, str(abs(num)))) for num in sequencia_atual]
    entradas = np.concatenate((diferenca, somas)).reshape(1, -1)
    
    ajustes = modelo.predict(entradas)
    ajustes = ajustes[0]
    proximos_numeros = np.array(sequencia_anterior) + ajustes
    proximos_numeros_arredondados = np.round(proximos_numeros).astype(int)
    proximos_numeros_ordenados = np.sort(proximos_numeros_arredondados)
    print(f"Números previstos (ordenados): {proximos_numeros_ordenados}")
    return proximos_numeros_ordenados

# Dados de exemplo
sequencia_atual = [4, 17, 19, 20, 40, 48]
sequencia_anterior = [15, 18, 27, 31, 39, 42]

# Mais sequências fictícias para treinamento
sequencias_atuais = [
    sequencia_atual,
    [6, 14, 22, 34, 36, 50],
    [8, 10, 21, 28, 37, 41]
]
sequencias_anteriores = [
    sequencia_anterior,
    [10, 16, 25, 32, 38, 45],
    [12, 14, 24, 30, 39, 43]
]
proximas_sequencias = [
    [24, 30, 43, 46, 55, 60],  # Exemplo de sequência real usada para treino
    [26, 29, 44, 47, 56, 62],
    [28, 32, 46, 50, 60, 64]
]

# Criando e treinando o modelo
modelo = criar_modelo()
treinar_modelo(modelo, sequencias_atuais, sequencias_anteriores, proximas_sequencias)

# Prevendo os próximos números
prever_proximos_numeros(modelo, sequencia_atual, sequencia_anterior)
