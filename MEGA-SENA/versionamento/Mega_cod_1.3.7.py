import os

# Configuração do ambiente TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# Função para carregar o CSV
def carregar_dados_csv():
    dados = pd.read_csv("MEGA-SENA/dados_megasena/Mega_Sena.csv", delimiter=';')
    dados.columns = dados.columns.str.strip()  # Remover espaços em branco
    return dados

# Função de cálculo LT
def calcular_lt(concurso_atual, concurso_referencia):
    previsao_lt = []
    for i in range(6):
        subtracao = concurso_atual[i] - concurso_referencia[i]
        soma_digitos_atual = sum(int(d) for d in str(concurso_atual[i]))
        soma_digitos_referencia = sum(int(d) for d in str(concurso_referencia[i]))
        soma_total = soma_digitos_atual + soma_digitos_referencia
        previsao = soma_total + subtracao
        if previsao < 0:
            previsao = abs(previsao)
        previsao_lt.append(previsao)
    return previsao_lt

# Função de cálculo reverso
def calcular_ajuste(previsao_lt, resultado_esperado):
    ajustes = []
    for i in range(6):
        ajuste = resultado_esperado[i] - previsao_lt[i]
        while ajuste > 9 or ajuste < -9:  # Ajustes compostos
            if ajuste > 9:
                ajuste -= 9
            elif ajuste < -9:
                ajuste += 9
        ajustes.append(ajuste)
    return ajustes

# Função para preparar os dados para treinamento
def preparar_dados_treinamento(dados):
    entradas = []
    saidas = []

    for i in range(6, len(dados) - 1):
        concurso_atual = dados.iloc[i][['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values
        concurso_atual = [int(x) for x in concurso_atual]

        num_concursos_referencia = concurso_atual[0]
        linha_referencia = i - num_concursos_referencia

        if linha_referencia < 0:
            continue

        concurso_referencia = dados.iloc[linha_referencia][['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values
        concurso_referencia = [int(x) for x in concurso_referencia]

        previsao_lt = calcular_lt(concurso_atual, concurso_referencia)

        resultado_esperado = dados.iloc[i + 1][['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values
        resultado_esperado = [int(x) for x in resultado_esperado]

        ajustes = calcular_ajuste(previsao_lt, resultado_esperado)

        entradas.append(previsao_lt)
        saidas.append(ajustes)

    return np.array(entradas), np.array(saidas)

# Função para construir o modelo
def construir_modelo():
    modelo = Sequential([
        Dense(64, input_dim=6, activation='relu'),
        Dense(64, activation='relu'),
        Dense(6, activation='linear')
    ])
    modelo.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return modelo

# Função para treinar o modelo
def treinar_modelo(modelo, entradas, saidas, dados):
    historico = modelo.fit(entradas, saidas, epochs=10, batch_size=16, verbose=0)

    for i in range(len(entradas)):
        previsao = modelo.predict(np.array([entradas[i]]), verbose=0)
        print(f"[INFO] Concurso Atual: {dados.iloc[6 + i][['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values}")
        print(f"[INFO] Previsão LT: {entradas[i]}")
        print(f"[INFO] Ajustes Esperados: {saidas[i]}")
        print(f"[INFO] Ajustes Calculados pela Rede Neural: {previsao[0]}")
        print("-")

    return historico

# Função principal
def main():
    dados = carregar_dados_csv()
    entradas, saidas = preparar_dados_treinamento(dados)

    print("[INFO] Dados preparados para treinamento.")
    print(f"[INFO] Total de exemplos: {len(entradas)}")

    modelo = construir_modelo()
    print("[INFO] Modelo construído.")

    print("[INFO] Iniciando treinamento.")
    historico = treinar_modelo(modelo, entradas, saidas, dados)

    modelo.save('modelo_ajustes.keras')
    print("[INFO] Modelo salvo como 'modelo_ajustes.keras'.")

if __name__ == "__main__":
    main()
