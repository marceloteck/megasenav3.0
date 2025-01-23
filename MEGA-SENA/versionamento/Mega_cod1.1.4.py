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
        Input(shape=(18,)),  # 6 diferenças + 12 somas
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
    return df.values

# Função para calcular a previsão ajustando com os números específicos
def calcular_previsao(seq_referencia, ajustes):
    melhores_ajustes = []
    ajustes_realizados = []  # Lista para armazenar os ajustes feitos
    for i, ajuste in enumerate(ajustes):
        previsao = seq_referencia + ajuste
        previsao = np.clip(previsao, 1, 60)  # Garantir que os números estejam entre 1 e 60
        diferenca = previsao - seq_referencia
        somas = [sum(map(int, str(abs(num)))) for num in seq_referencia] + \
                [sum(map(int, str(abs(num)))) for num in previsao]
        entrada = np.concatenate((diferenca, somas))
        melhores_ajustes.append(entrada)

        # Imprimir o ajuste feito para cada número da sequência
        ajustes_realizados.append((seq_referencia, previsao, ajuste))
    
    # Imprimir ajustes realizados
    print("Ajustes realizados para a sequência de referência:", seq_referencia)
    for seq_ref, previsao, ajuste in ajustes_realizados:
        print(f"Ajuste {ajuste}: {seq_ref} -> {previsao}")
    
    return np.array(melhores_ajustes)


# Função para realizar o treinamento supervisionado
def treinamento_supervisionado_com_calculos(modelo, sequencias, start_line=15):
    ajustes = np.array([1, -1, 2, -2, 3, -3, 6, -6, 9, -9])

    for i in range(start_line, len(sequencias) - 1):
        seq_atual = sequencias[i]
        seq_proxima = sequencias[i + 1]
        num_linhas_voltar = seq_atual[0]

        if i - num_linhas_voltar < 0:
            print(f"Não há linhas suficientes para calcular para a sequência {seq_atual}. Pulando.")
            continue

        seq_referencia = sequencias[i - num_linhas_voltar]
        entradas_ajustadas = calcular_previsao(seq_referencia, ajustes)
        previsoes = modelo.predict(entradas_ajustadas)
        previsao_final = np.round(previsoes).astype(int).flatten()
        previsao_final = np.clip(previsao_final, 1, 60)

        print(f"\nSequência atual da planilha (linha {i + 1}): {seq_atual}")
        print(f"Sequência de referência (linha {i + 1 - num_linhas_voltar}): {seq_referencia}")
        print(f"Previsão final feita pelo modelo: {previsao_final[:6]}")
        print(f"Sequência real da próxima linha (linha {i + 2}): {seq_proxima}")

        feedback = input("A previsão está correta? (s/n): ").strip().lower()
        if feedback == 'n':
            sequencia_correta = input("Por favor, insira a sequência correta (separada por espaços): ")
            sequencia_correta = np.array([int(num) for num in sequencia_correta.split()])
            modelo.fit(entradas_ajustadas, sequencia_correta.reshape(1, -1), epochs=10, verbose=0)
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
sequencias = carregar_dados(filepath)

# Carregando ou criando o modelo
modelo = carregar_ou_criar_modelo()

# Realizando o treinamento supervisionado com cálculos exibidos
treinamento_supervisionado_com_calculos(modelo, sequencias)
