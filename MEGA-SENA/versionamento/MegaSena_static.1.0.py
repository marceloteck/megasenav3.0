import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Configuração do ambiente TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 1. Carregamento de Dados
def carregar_dados_csv(caminho_arquivo):
    dados = pd.read_csv(caminho_arquivo, sep=";")
    return dados[['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values

# 2. Cálculo LT
def calcular_LT(sequencia_atual, sequencia_referencia):
    resultado_LT = []
    for num_atual, num_ref in zip(sequencia_atual, sequencia_referencia):
        subtracao = num_atual - num_ref
        soma_digitos = sum(map(int, str(num_atual))) + sum(map(int, str(num_ref)))
        ajuste = subtracao + soma_digitos

        # Ajuste final (dentro do intervalo 1-60)
        while ajuste < 1 or ajuste > 60:
            ajuste = ajuste - 60 if ajuste > 60 else ajuste + 60
        resultado_LT.append(ajuste)

    return resultado_LT

# 3. Análise Estatística e Probabilidade
def calcular_probabilidades(dados):
    frequencias = np.zeros(60)
    for linha in dados:
        for numero in linha:
            frequencias[numero - 1] += 1
    probabilidades = frequencias / frequencias.sum()
    return probabilidades

# 4. Treinamento da Rede Neural
def criar_modelo():
    modelo = Sequential([
        Dense(64, activation='relu', input_shape=(12,)),
        Dense(32, activation='relu'),
        Dense(6, activation='linear')  # Saída com 6 números previstos
    ])
    modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return modelo

def treinar_modelo(dados, modelo):
    X_treino = []  # Features: sequência atual + referência
    y_treino = []  # Labels: sequência esperada

    for i in range(len(dados) - 1):
        sequencia_atual = dados[i]
        sequencia_referencia = dados[i + 1]
        sequencia_esperada = dados[i + 2] if i + 2 < len(dados) else dados[0]

        X_treino.append(np.concatenate([sequencia_atual, sequencia_referencia]))
        y_treino.append(sequencia_esperada)

    X_treino, y_treino = np.array(X_treino), np.array(y_treino)
    modelo.fit(X_treino, y_treino, epochs=50, batch_size=16, verbose=1)

# 5. Avaliação do Modelo
def avaliar_modelo(modelo, dados):
    acertos_totais = 0
    total_tentativas = 0

    for i in range(len(dados) - 2):
        sequencia_atual = dados[i]
        sequencia_referencia = dados[i + 1]
        sequencia_esperada = dados[i + 2]

        entrada = np.concatenate([sequencia_atual, sequencia_referencia]).reshape(1, -1)
        previsao = modelo.predict(entrada)
        previsao_ajustada = [int(round(num)) for num in previsao[0]]

        # Contar acertos
        acertos = len(set(previsao_ajustada) & set(sequencia_esperada))
        acertos_totais += acertos
        total_tentativas += 6

        print(f"Sequência esperada: {sequencia_esperada}")
        print(f"Previsão: {previsao_ajustada}")
        print(f"Acertos: {acertos}/6\n")

    print(f"Taxa de acerto total: {acertos_totais / total_tentativas * 100:.2f}%")

# Fluxo Principal
def main():
    caminho_arquivo = 'MEGA-SENA/dados_megasena/Mega_Sena.1.0.csv'
    dados = carregar_dados_csv(caminho_arquivo)

    print("Calculando probabilidades...")
    probabilidades = calcular_probabilidades(dados)
    print(f"Probabilidades calculadas: {probabilidades[:10]}...")

    print("Criando modelo...")
    modelo = criar_modelo()

    print("Treinando modelo...")
    treinar_modelo(dados, modelo)

    print("Avaliando modelo...")
    avaliar_modelo(modelo, dados)

if __name__ == '__main__':
    main()
