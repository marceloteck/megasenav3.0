import os
# Configurações do ambiente
os.system('title Mega Código 2.0')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt




# Função para criar o modelo de rede neural
def criar_modelo():
    modelo = Sequential([
        Input(shape=(18,)),  # 6 diferenças + 6 somas + 6 somas da outra sequência
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(6)  # Saída com 6 ajustes numéricos
    ])
    modelo.compile(optimizer='adam', loss='mse')
    return modelo

# Função para carregar e preprocessar os dados
def carregar_dados(filepath):
    # Carregar o CSV
    df = pd.read_csv(filepath, sep=';')

    # Ordenar pelo tempo, se houver uma coluna de data
    if 'Data' in df.columns:  # Supondo que a coluna de data seja 'Data'
        df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y')  # Converter para datetime
        df.sort_values('Data', inplace=True)  # Ordenar pelo tempo
        df.reset_index(drop=True, inplace=True)  # Resetar o índice após a ordenação

    # Selecionar as colunas de números
    df = df[['N1', 'N2', 'N3', 'N4', 'N5', 'N6']]
    sequencias = df.values
    print(sequencias)
    entradas, saidas = [], []
    for i in range(1, len(sequencias)):
        seq_atual = sequencias[i - 1]
        seq_prox = sequencias[i]
        diferenca = seq_prox - seq_atual
        somas = [sum(map(int, str(abs(num)))) for num in seq_atual] + \
                [sum(map(int, str(abs(num)))) for num in seq_prox]
        entradas.append(np.concatenate((diferenca, somas)))
        saidas.append(diferenca)  # Ajustes para todos os números

    return np.array(entradas), np.array(saidas)


# Função para treinar o modelo
def treinar_modelo(modelo, entradas_treino, saidas_treino):
    modelo.fit(entradas_treino, saidas_treino, epochs=50, verbose=1)  # Treinamento com 500 épocas

# Função para avaliar o modelo
def avaliar_modelo(modelo, entradas_teste, saidas_teste):
    loss = modelo.evaluate(entradas_teste, saidas_teste, verbose=0)
    print(f'Erro médio quadrático no conjunto de teste: {loss}')


# Função para prever os próximos números
def prever_proximos_numeros(modelo, sequencia_atual, sequencia_anterior):
    diferenca = np.array(sequencia_anterior) - np.array(sequencia_atual)
    somas = [sum(map(int, str(abs(num)))) for num in sequencia_anterior] + \
            [sum(map(int, str(abs(num)))) for num in sequencia_atual]
    entradas = np.concatenate((diferenca, somas)).reshape(1, -1)
    
    ajustes = modelo.predict(entradas)
    proxima_sequencia = np.array(sequencia_anterior) + ajustes[0]
    proxima_sequencia_arredondada = np.round(proxima_sequencia).astype(int)
    
    # Remover números repetidos
    proxima_sequencia_unica = np.unique(proxima_sequencia_arredondada)
    
    # Garantir que há 6 números (caso precise ajustar para completar a sequência)
    if len(proxima_sequencia_unica) < 6:
        proxima_sequencia_unica = np.pad(proxima_sequencia_unica, (0, 6 - len(proxima_sequencia_unica)), mode='wrap')

    proxima_sequencia_ordenada = np.sort(proxima_sequencia_unica)
    print(f"Números previstos: {proxima_sequencia_ordenada}")
    return proxima_sequencia_ordenada

    
    
def avaliar_desempenho(numeros_reais, numeros_previstos):
    # 1. Gráfico de Dispersão (Erro de Previsão vs Real)
    plt.figure(figsize=(12, 10))

    plt.subplot(3, 2, 1)
    plt.scatter(numeros_reais, numeros_previstos, color='blue', marker='o')
    plt.plot([min(numeros_reais), max(numeros_reais)], [min(numeros_reais), max(numeros_reais)], color='red', linestyle='--')
    plt.title('Dispersão: Números Reais vs Números Previstos')
    plt.xlabel('Números Reais')
    plt.ylabel('Números Previstos')

    # 2. Histograma do Erro Absoluto
    erro = np.abs(numeros_reais - numeros_previstos)
    plt.subplot(3, 2, 2)
    plt.hist(erro, bins=10, color='green', edgecolor='black', alpha=0.7)
    plt.title('Histograma do Erro Absoluto')
    plt.xlabel('Erro Absoluto')
    plt.ylabel('Frequência')

    # 3. Gráfico de Correlação
    plt.subplot(3, 2, 3)
    plt.scatter(numeros_reais, numeros_previstos, color='purple', marker='x')
    plt.title('Correlação: Números Reais vs Números Previstos')
    plt.xlabel('Números Reais')
    plt.ylabel('Números Previstos')

    # 4. Gráfico de Linha de Erro Absoluto ao Longo do Tempo
    plt.subplot(3, 2, 4)
    plt.plot(erro, marker='o', linestyle='-', color='orange')
    plt.title('Erro Absoluto ao Longo do Tempo')
    plt.xlabel('Índice')
    plt.ylabel('Erro Absoluto')

    # 5. Boxplot do Erro Absoluto
    plt.subplot(3, 2, 5)
    plt.boxplot(erro, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.title('Boxplot do Erro Absoluto')
    plt.xlabel('Erro Absoluto')

    # 6. Gráfico de Desempenho Comparativo (Reais vs Previstos)
    plt.subplot(3, 2, 6)
    plt.plot(numeros_reais, label='Reais', marker='o', linestyle='-', color='blue')
    plt.plot(numeros_previstos, label='Previstos', marker='x', linestyle='--', color='red')
    plt.title('Desempenho Comparativo: Reais vs Previstos')
    plt.xlabel('Índice')
    plt.ylabel('Números')
    plt.legend()

    plt.tight_layout()
    plt.show()
    

# Caminho para o arquivo CSV
filepath = 'MEGA-SENA/dados_megasena/Mega_Sena.csv'

# Carregando e preprocessando os dados
entradas, saidas = carregar_dados(filepath)

# Dividindo os dados em treinamento e teste
entradas_treino, entradas_teste, saidas_treino, saidas_teste = train_test_split(entradas, saidas, test_size=0.2, random_state=42)

# Criando e treinando o modelo
modelo = criar_modelo()
treinar_modelo(modelo, entradas_treino, saidas_treino)

# Avaliando o modelo
avaliar_modelo(modelo, entradas_teste, saidas_teste)

# Prevendo os próximos números para a última sequência
numeros_previstos = prever_proximos_numeros(modelo, entradas[-1][:6], entradas[-1][6:12])


# Plotando os gráficos de análise
avaliar_desempenho(saidas[-1], numeros_previstos)