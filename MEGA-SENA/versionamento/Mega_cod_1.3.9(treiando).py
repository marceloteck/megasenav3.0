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
    dados = pd.read_csv("MEGA-SENA/dados_megasena/Mega_Sena.1.0.csv", delimiter=';')
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
        ajuste = resultado_esperado[i] - previsao_lt[i]  # Diferença entre o resultado esperado e o cálculo LT
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

def arredondar_ajustes(previsao):
    return np.round(previsao).astype(int)

# Função para construir o modelo

def construir_modelo():
    modelo = Sequential([
        Dense(64, input_dim=6, activation='relu'),
        Dense(64, activation='relu'),
        Dense(6, activation='linear')
    ])
    modelo.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return modelo
    



# Função de treinamento básica com cálculo LT e ajustes esperados
def treinar_modelo_basico(modelo, entradas, saidas, dados):
    acertos_totais = {6: 0, 5: 0, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0}
    
    for i in range(len(entradas)):
        concurso_atual = dados.iloc[6 + i][['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values
        concurso_referencia = dados.iloc[6 + i - 1][['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values
        resultado_esperado = dados.iloc[6 + i + 1][['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values

        # Calcular LT
        previsao_lt = calcular_lt(concurso_atual, concurso_referencia)
        # Calcular Ajuste
        ajustes = calcular_ajuste(previsao_lt, resultado_esperado)

        # Resultado LT + Ajuste Esperado (Resultado LT Plus)
        resultado_lt_plus = [int(previsao_lt[j] + ajustes[j]) for j in range(6)]

        # Verificar se o Resultado LT Plus é igual ao Resultado Esperado
        acerto = sum(1 for a, b in zip(resultado_lt_plus, resultado_esperado) if a == b)
        
        # Atualizar contador de acertos
        acertos_totais[acerto] += 1

        # Exibição
        print(f"[INFO] Concurso Atual (Linha {6 + i}): {list(map(int, concurso_atual))}")
        print(f"[INFO] Concurso Referência (Linha {6 + i - 1}): {list(map(int, concurso_referencia))}")
        print(f"[INFO] Resultado Esperado: {list(map(int, resultado_esperado))}")
        print(f"[INFO] Cálculo LT: {list(map(int, previsao_lt))}")  # Convertendo np.int64 para int
        print(f"[INFO] Ajustes Esperados: {list(map(int, ajustes))}")  # Convertendo np.int64 para int
        print(f"[INFO] Resultado LT Plus: {resultado_lt_plus}")
        print(f"[INFO] Acertos: {acerto} acertos")
        print("-" * 50)

        # Treinamento da rede neural - Aqui a rede vai aprender com os dados
        modelo.fit(entradas, saidas, epochs=1, verbose=0)

    # Relatório Final de Acertos
    print("\n[INFO] Relatório de Acertos Final:")
    for acertos, quantidade in acertos_totais.items():
        print(f"{acertos} acertos: {quantidade} vez(es)")



        
        
        
        
        




# Função principal
def main():
    # Carregar dados
    dados = carregar_dados_csv()
    entradas, saidas = preparar_dados_treinamento(dados)

    print("[INFO] Dados preparados para treinamento.")
    print(f"[INFO] Total de exemplos: {len(entradas)}")

    # Criar o modelo
    modelo = construir_modelo()
    print("[INFO] Modelo construído.")

    # Treinar o modelo
    treinar_modelo_basico(modelo, entradas, saidas, dados)

    # Salvar o modelo treinado
    modelo.save('modelo_ajustes.keras')
    print("[INFO] Modelo salvo como 'modelo_ajustes.keras'.")

if __name__ == "__main__":
    main()