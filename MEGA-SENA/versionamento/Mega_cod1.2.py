import os

# Configurações do ambiente
os.system('title Mega Código 1.2')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input



# Função para carregar os dados do CSVw
def carregar_dados(filepath):
    dados = pd.read_csv(filepath,  sep=';')
    sequencias = dados[['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values
    return sequencias

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

# Função para preparar os dados de entrada e saída para o modelo
def preparar_dados(sequencias):
    entradas = []
    saídas = []
    for i in range(len(sequencias) - 2):
        seq_atual = sequencias[i]
        seq_ant = sequencias[i + 1]
        seq_prox = sequencias[i + 2]
        
        diferenca = seq_ant - seq_atual
        somas = [sum(map(int, str(abs(num)))) for num in seq_ant] + \
                [sum(map(int, str(abs(num)))) for num in seq_atual]
        entradas.append(np.concatenate((diferenca, somas)))
        saídas.append(seq_prox[0] - seq_ant[0])  # Ajuste esperado para o primeiro número
    
    return np.array(entradas), np.array(saídas)

# Função para treinar o modelo
def treinar_modelo(modelo, entradas, saídas):
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

# Carregar os dados do CSV
filepath = 'MEGA-SENA/dados_megasena/Mega_Sena.csv'
sequencias = carregar_dados(filepath)

# Preparar os dados para o modelo
entradas, saídas = preparar_dados(sequencias)

# Criando e treinando o modelo
modelo = criar_modelo()
treinar_modelo(modelo, entradas, saídas)

# Prevendo o próximo número com as duas últimas sequências
prever_proximo_numero(modelo, sequencias[-2], sequencias[-3])
