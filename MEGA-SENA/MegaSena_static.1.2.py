import os

# Configuração do ambiente TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split


# Caminho do arquivo CSV e do modelo salvo
data_path = 'MEGA-SENA/dados_megasena/Mega_Sena.1.0.csv'
model_path = 'modelo_megasena.keras'

# Função para carregar dados do CSV
def carregar_dados_csv(caminho):
    dados = pd.read_csv(caminho, sep=";")
    return dados[['N1', 'N2', 'N3', 'N4', 'N5', 'N6']]

# Função para realizar o cálculo LT
def calcular_lt(sequencia1, sequencia2):
    resultado = []
    for num1, num2 in zip(sequencia1, sequencia2):
        subtracao = num1 - num2
        soma_digitos1 = sum(map(int, str(num1)))
        soma_digitos2 = sum(map(int, str(num2)))
        soma_total = soma_digitos1 + soma_digitos2 + subtracao
        resultado.append(soma_total % 60 if soma_total % 60 != 0 else 60)
    return resultado

# Função para ajustar resultados com base nos ajustes aprendidos pelo modelo
def aplicar_ajustes(resultados, ajustes):
    ajustados = []
    for res, ajuste in zip(resultados, ajustes):
        valor_ajustado = res + ajuste
        while valor_ajustado > 60:
            valor_ajustado -= 60
        while valor_ajustado < 1:
            valor_ajustado += 60
        ajustados.append(valor_ajustado)
    return ajustados

# Preparação dos dados
def preparar_dados(dados):
    sequencias = dados.values
    entradas = []
    saidas = []
    for i in range(len(sequencias) - 2):
        entrada = calcular_lt(sequencias[i], sequencias[i + 1])
        saida = sequencias[i + 2]
        entradas.append(entrada)
        saidas.append(saida)
    return np.array(entradas), np.array(saidas)

# Função para criar e compilar o modelo
def criar_modelo():
    modelo = Sequential([
        Dense(256, activation='relu', input_dim=6),
        Dropout(2.0),
        Dense(128, activation='relu'),
        Dropout(2.0),
        Dense(64, activation='relu'),
        Dense(6, activation='linear')
    ])
    modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return modelo

# Loop de treinamento e avaliação
def treinar_e_prever_em_loop(max_tentativas=10, erro_maximo=5.0):
    # Carregar os dados
    dados = carregar_dados_csv(data_path)
    entradas, saidas = preparar_dados(dados)

    # Dividir dados em treinamento e validação
    X_train, X_val, y_train, y_val = train_test_split(entradas, saidas, test_size=0.2, random_state=42)

    # Carregar modelo salvo ou criar um novo
    if os.path.exists(model_path):
        modelo = load_model(model_path)
        print("Modelo carregado com sucesso!")
    else:
        modelo = criar_modelo()
        print("Novo modelo criado.")

    tentativa = 0
    while tentativa < max_tentativas:
        print(f"\nIniciando tentativa {tentativa + 1}/{max_tentativas}...")
        # Treinar o modelo
        modelo.fit(X_train, y_train, epochs=10, batch_size=12, validation_data=(X_val, y_val), verbose=1)

        # Avaliar o modelo
        perda, mae = modelo.evaluate(X_val, y_val, verbose=0)
        print(f"Erro médio absoluto na validação (MAE): {mae:.2f}")

        # Verificar o erro
        if mae <= erro_maximo:
            print("Erro aceitável atingido. Salvando o modelo...")
            modelo.save(model_path)
            break
        else:
            print("Erro ainda alto. Continuando o treinamento...")
            modelo.save(model_path)

        tentativa += 1

    # Prever a próxima sequência com base no último concurso
    ultima_sequencia = calcular_lt(dados.values[-2], dados.values[-1])
    previsao = modelo.predict(np.array([ultima_sequencia]))[0]

    # Ajustar previsão para ficar no intervalo de 1 a 60 e ordenar em ordem crescente
    previsao_ajustada = sorted(aplicar_ajustes(previsao, [0, 0, 0, 0, 0, 0]))

    print("\nPróxima sequência prevista:", [round(num) for num in previsao_ajustada])

# Executar o processo de treinamento e previsão em loop
treinar_e_prever_em_loop()
