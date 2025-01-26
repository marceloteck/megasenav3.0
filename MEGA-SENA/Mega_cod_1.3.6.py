# Configuração do ambiente TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam



# Função para carregar o CSV
def carregar_dados_csv():
    dados = pd.read_csv("MEGA-SENA/dados_megasena/Mega_Sena.1.0.csv", delimiter=';')
    dados.columns = dados.columns.str.strip()  # Remover espaços em branco das colunas
    return dados

# Função para o cálculo LT
def calcular_lt(concurso_atual, concurso_referencia):
    previsao_lt = []
    for i in range(6):
        subtracao = concurso_atual[i] - concurso_referencia[i]
        soma_digitos_atual = sum(int(d) for d in str(concurso_atual[i]))
        soma_digitos_ref = sum(int(d) for d in str(concurso_referencia[i]))
        soma_total = soma_digitos_atual + soma_digitos_ref + subtracao
        previsao_lt.append(soma_total)
    return previsao_lt

# Função para o cálculo reverso
def calcular_ajustes_reversos(previsao, resultado_esperado):
    ajustes = []
    for p, r in zip(previsao, resultado_esperado):
        ajuste_total = 0
        while p != r:
            if p < r:
                if r - p >= 9:
                    p += 9
                    ajuste_total += 9
                elif r - p >= 6:
                    p += 6
                    ajuste_total += 6
                elif r - p >= 3:
                    p += 3
                    ajuste_total += 3
                else:
                    p += 1
                    ajuste_total += 1
            else:
                if p - r >= 9:
                    p -= 9
                    ajuste_total -= 9
                elif p - r >= 6:
                    p -= 6
                    ajuste_total -= 6
                elif p - r >= 3:
                    p -= 3
                    ajuste_total -= 3
                else:
                    p -= 1
                    ajuste_total -= 1
        ajustes.append(ajuste_total)
    return ajustes

# Preparar os dados para treinamento da rede neural
def preparar_dados_treinamento(dados):
    X = []  # Entradas: Previsão LT
    y = []  # Saídas: Ajustes necessários

    for i in range(6, len(dados) - 1):
        concurso_atual = dados.iloc[i][['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values.astype(int)
        num_concursos_referencia = concurso_atual[0]
        linha_referencia = i - num_concursos_referencia

        if linha_referencia < 0:
            continue

        concurso_referencia = dados.iloc[linha_referencia][['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values.astype(int)
        resultado_esperado = dados.iloc[i + 1][['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values.astype(int)

        previsao_lt = calcular_lt(concurso_atual, concurso_referencia)
        ajustes = calcular_ajustes_reversos(previsao_lt, resultado_esperado)

        X.append(previsao_lt)
        y.append(ajustes)

    return np.array(X), np.array(y)

# Criar a rede neural para ajustes
def criar_rede_neural():
    modelo = Sequential([
        Dense(64, activation='relu', input_shape=(6,)),
        Dense(32, activation='relu'),
        Dense(6, activation='linear')  # Saída com 6 ajustes
    ])
    modelo.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return modelo

# Função principal
def main():
    dados = carregar_dados_csv()

    # Preparar dados de treinamento
    X, y = preparar_dados_treinamento(dados)
    print(f"[INFO] Dados de treinamento prontos. {len(X)} exemplos preparados.")

    # Criar e treinar a rede neural
    modelo = criar_rede_neural()
    modelo.fit(X, y, epochs=50, batch_size=16, validation_split=0.2)

    # Salvar o modelo treinado
    modelo.save("modelo_ajustes.keras")
    print("[INFO] Modelo treinado e salvo como 'modelo_ajustes.keras'.")

if __name__ == "__main__":
    main()
