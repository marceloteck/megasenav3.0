import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input

# Configurações do ambiente
os.system('title Mega Código 2.0')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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

# Função para carregar o modelo existente ou criar um novo
def carregar_ou_criar_modelo():
    if os.path.exists('modelo_megasena.keras'):
        return load_model('modelo_megasena.keras')
    else:
        return criar_modelo()

# Função para carregar e preprocessar os dados
def carregar_dados(filepath):
    df = pd.read_csv(filepath, sep=';') 
    df = df[['N1', 'N2', 'N3', 'N4', 'N5', 'N6']]
    sequencias = df.values
    entradas, saidas = [], []
    for i in range(1, len(sequencias)):
        seq_atual = sequencias[i - 1]
        seq_prox = sequencias[i]
        diferenca = seq_prox - seq_atual
        somas = [sum(map(int, str(abs(num)))) for num in seq_atual] + \
                [sum(map(int, str(abs(num)))) for num in seq_prox]
        entradas.append(np.concatenate((diferenca, somas)))
        saidas.append(diferenca)
    return np.array(entradas), np.array(saidas), sequencias

# Função para realizar o treinamento supervisionado
def treinamento_supervisionado_com_calculos(modelo, entradas, saidas, sequencias):
    for i in range(len(entradas)):
        seq_atual = sequencias[i]
        seq_prox = sequencias[i + 1]

        print(f"\nSequência atual da planilha: {seq_atual}")
        print(f"Próxima sequência da planilha: {seq_prox}")
        
        diferenca = entradas[i][:6]
        somas = entradas[i][6:]
        print(f"Diferença entre sequências: {diferenca}")
        print(f"Somas dos dígitos (sequência anterior e atual): {somas[:6]} e {somas[6:]}")

        previsao = modelo.predict(entradas[i].reshape(1, -1))
        previsao_arredondada = np.round(previsao).astype(int).flatten()
        
        print(f"\nPrevisão feita pelo modelo: {previsao_arredondada}")
        print(f"Sequência real (diferença): {saidas[i]}")
        
        feedback = input("A previsão está correta? (s/n): ").strip().lower()
        if feedback == 'n':
            sequencia_correta = input("Por favor, insira a sequência correta (separada por espaços): ")
            sequencia_correta = np.array([int(num) for num in sequencia_correta.split()])
            modelo.fit(entradas[i].reshape(1, -1), sequencia_correta.reshape(1, -1), epochs=10, verbose=0)
            print("Re-treinamento concluído.")
            modelo.save('modelo_megasena.keras')
        elif feedback == 's':
            print("Previsão aceita. Modelo salvo.")
            modelo.save('modelo_megasena.keras')
        else:
            print("Entrada inválida, continuando para a próxima previsão.")

# Caminho para o arquivo CSV
filepath = 'MEGA-SENA/dados_megasena/Mega_Sena.csv'

# Carregando e preprocessando os dados
entradas, saidas, sequencias = carregar_dados(filepath)

# Carregando ou criando o modelo
modelo = carregar_ou_criar_modelo()

# Realizando o treinamento supervisionado com cálculos exibidos
treinamento_supervisionado_com_calculos(modelo, entradas, saidas, sequencias)
