import os

# Configurações do ambiente
os.system('title Mega Código 1.0')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np


# Dados de exemplo
sequencia_atual = [4, 17, 19, 20, 40, 48]
sequencia_anterior = [15, 18, 27, 31, 39, 42]
proxima_sequencia = [24, 30, 43, 46, 55, 60]

# Função para criar o modelo de rede neural
def criar_modelo():
    modelo = Sequential([
        Input(shape=(18,)),  # 6 diferenças + 6 somas + 6 somas da outra sequência
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)  # Saída de um ajuste numérico
    ])
    modelo.compile(optimizer='adam', loss='mse')
    return modelo

# Função para treinar o modelo
def treinar_modelo(modelo, sequencias_atuais, sequencias_anteriores, proximas_sequencias):
    entradas = []
    saídas = []
    for seq_atual, seq_ant, seq_prox in zip(sequencias_atuais, sequencias_anteriores, proximas_sequencias):
        diferenca = np.array(seq_ant) - np.array(seq_atual)
        somas = [sum(map(int, str(abs(num)))) for num in seq_ant] + \
                [sum(map(int, str(abs(num)))) for num in seq_atual]
        entradas.append(np.concatenate((diferenca, somas)))
        saídas.append(seq_prox[0] - seq_ant[0])  # Ajuste esperado para o primeiro número

    entradas = np.array(entradas)
    saídas = np.array(saídas)
    modelo.fit(entradas, saídas, epochs=500, verbose=0)  # Treinamento com 500 épocas

# Função para prever o próximo número
def prever_proximo_numero(modelo, sequencia_atual, sequencia_anterior):
    diferenca = np.array(sequencia_anterior) - np.array(sequencia_atual)
    somas = [sum(map(int, str(abs(num)))) for num in sequencia_anterior] + \
            [sum(map(int, str(abs(num)))) for num in sequencia_atual]
    entradas = np.concatenate((diferenca, somas)).reshape(1, -1)
    
    ajuste = modelo.predict(entradas)
    ajuste = ajuste[0][0]
    print(f"Ajuste previsto para o próximo número: {ajuste}")

    proximo_numero = sequencia_anterior[0] + ajuste
    proximo_numero_arredondado = round(proximo_numero)
    print(f"Número previsto: {proximo_numero_arredondado}")

# Criando e treinando o modelo
modelo = criar_modelo()
treinar_modelo(modelo, [sequencia_atual], [sequencia_anterior], [proxima_sequencia])

# Prevendo o próximo número
prever_proximo_numero(modelo, sequencia_atual, sequencia_anterior)
