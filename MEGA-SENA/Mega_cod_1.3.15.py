import os

# Configuração do ambiente TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import time

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
        ajuste = resultado_esperado[i] - previsao_lt[i]
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
        Dense(256, input_dim=6, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(6, activation='linear')
    ])
    modelo.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='mse', metrics=['accuracy'])
    return modelo

def ajustar_intervalo(previsao):
    # Limita os valores no intervalo de 1 a 60
    return [max(1, min(60, int(p))) for p in previsao]

# Função de treinamento com cálculo LT e comparação com a rede neural
def treinar_modelo_basico(modelo, entradas, saidas, dados):
    acertos_lt_totais = {6: 0, 5: 0, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0}
    acertos_rede_totais = {6: 0, 5: 0, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0}

    historico_perda = []

    entradas_normalizadas = entradas #/ 60.0

    for i in range(len(entradas)):
        L_ATUAL = 6 + i
        L_ESPER = L_ATUAL + 1
        concurso_atual = dados.iloc[L_ATUAL][['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values
        L_REFER = L_ATUAL - concurso_atual[0]
        concurso_referencia = dados.iloc[L_REFER][['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values
        resultado_esperado = dados.iloc[L_ESPER][['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values

        previsao_lt = calcular_lt(concurso_atual, concurso_referencia)
        ajustes = calcular_ajuste(previsao_lt, resultado_esperado)
        resultado_lt_plus = [int(previsao_lt[j] + ajustes[j]) for j in range(6)]

        acerto_lt = sum(1 for a, b in zip(resultado_lt_plus, resultado_esperado) if a == b)
        acertos_lt_totais[acerto_lt] += 1

        # Adiciona o treinamento com EarlyStopping
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        historico = modelo.fit(entradas_normalizadas, saidas, batch_size=16, epochs=1, callbacks=[early_stopping], verbose=0)
        historico_perda.append(historico.history['loss'][0])

        previsao_rede = modelo.predict(np.array([entradas_normalizadas[i]]), verbose=0)
        previsao_rede = arredondar_ajustes(previsao_rede[0])
        
        # Agora, ajusta os valores para o intervalo de 1 a 60
        previsao_rede = ajustar_intervalo(previsao_rede)

        acertos_rede = sum(1 for a, b in zip(ajustes, previsao_rede) if a == b)
        acertos_rede_totais[acertos_rede] += 1
        resultado_rd_plus = [int(previsao_lt[j] + previsao_rede[j]) for j in range(6)]

        # Mostra as informações de cada iteração
        print(f"\n[INFO] Concurso Referência (Linha {L_REFER}): {list(map(int, concurso_referencia))}")
        print(f"[INFO] Concurso Atual (Linha {L_ATUAL}): {list(map(int, concurso_atual))}")
        print(f"[INFO] Resultado Esperado (Linha {L_ESPER}): {list(map(int, resultado_esperado))}\n")
        print(f"[INFO] Cálculo LT: {list(map(int, previsao_lt))}")
        print(f"[INFO] Ajustes Esperados: {list(map(int, ajustes))}")
        print(f"[INFO] Resultado LT + AJUSTES: {list(map(int, resultado_lt_plus))}")
        print(f"[INFO] Acertos do Cálculo LT: {acerto_lt} acertos\n")
        print(f"[INFO] Previsão do AJUSTE da Rede Neural: {list(map(int, previsao_rede))}")
        print(f"[INFO] Previsão da próx. sequência Rede Neural: {list(map(int, resultado_rd_plus))}")
        print(f"[INFO] Acertos da Rede Neural: {acertos_rede} acertos")
        print("-" * 50)

        # Verifica se acertou os 6 números
        while acertos_rede < 6:
            print("[INFO] Rede neural não acertou todos os números, realizando novo treinamento...")
            start_time = time.time()  # Começa a medir o tempo

            # Treina novamente a rede neural e faz a previsão
            historico = modelo.fit(entradas_normalizadas, saidas, batch_size=16, epochs=1, callbacks=[early_stopping], verbose=0)
            previsao_rede = modelo.predict(np.array([entradas_normalizadas[i]]), verbose=0)
            previsao_rede = arredondar_ajustes(previsao_rede[0])
            previsao_rede = ajustar_intervalo(previsao_rede)

            acertos_rede = sum(1 for a, b in zip(ajustes, previsao_rede) if a == b)
            resultado_rd_plus = [int(previsao_lt[j] + previsao_rede[j]) for j in range(6)]

            # Exibe o tempo gasto para essa iteração
            elapsed_time = time.time() - start_time
            print(f"[INFO] Tempo gasto para essa iteração: {elapsed_time:.2f} segundos")

            # Mostra novamente as informações de cada iteração
            print(f"[INFO] Previsão do AJUSTE da Rede Neural: {list(map(int, previsao_rede))}")
            print(f"[INFO] Previsão da próx. sequência Rede Neural: {list(map(int, resultado_rd_plus))}")
            print(f"[INFO] Ajustes Esperados: {list(map(int, ajustes))}")
            print(f"[INFO] Acertos da Rede Neural: {acertos_rede} acertos\n")

        print("[INFO] Rede neural acertou todos os 6 números! Passando para a próxima análise...\n")

    print("\n[INFO] Relatório Final de Acertos do Cálculo LT:")
    for acertos, quantidade in acertos_lt_totais.items():
        print(f"{acertos} acertos: {quantidade} vez(es)")

    print("\n[INFO] Relatório Final de Acertos da Rede Neural:")
    for acertos, quantidade in acertos_rede_totais.items():
        print(f"{acertos} acertos: {quantidade} vez(es)")



 # Previsão do próximo concurso
    print("\n[INFO] Fazendo previsão para o próximo concurso da Mega-Sena...")
    ultima_entrada = entradas_normalizadas[-1].reshape(1, -1)  # Usar a última entrada para previsão
    previsao_rede = modelo.predict(ultima_entrada, verbose=0)
    previsao_rede = arredondar_ajustes(previsao_rede[0])
    previsao_rede = ajustar_intervalo(previsao_rede)

    # Exibir concursos atual e referência e os ajustes usados na previsão
    concurso_atual = dados.iloc[-1][['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values
    REFERENCIA = concurso_atual[0]
    
    concurso_referencia = dados.iloc[-(REFERENCIA)][['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values
    
    previsao_lt = calcular_lt(concurso_atual, concurso_referencia)
    ajustes = calcular_ajuste(previsao_lt, previsao_rede)

    # Exibir os resultados
    print("\n\n")
    print(f"[INFO] Concurso Atual (Último): {list(map(int, concurso_atual))}")
    print(f"[INFO] Concurso de Referência ({REFERENCIA}): {list(map(int, concurso_referencia))}")
    print(f"[INFO] Cálculo LT: {list(map(int, previsao_lt))}")
    print(f"[INFO] Ajustes usados para a previsão: {list(map(int, ajustes))}")
    print(f"[INFO] Previsão para o próximo concurso: {previsao_rede}")
    
    import matplotlib.pyplot as plt
    plt.plot(historico_perda)
    plt.title("Histórico de Perda do Treinamento")
    plt.xlabel("Épocas")
    plt.ylabel("Perda (Loss)")
    plt.show()
    
    


# Função para carregar o modelo salvo
def carregar_modelo_salvo(caminho_modelo):
    try:
        modelo = load_model(caminho_modelo)
        print(f"[INFO] Modelo carregado de {caminho_modelo}.")
    except:
        modelo = construir_modelo()  # Caso não encontre o modelo salvo, cria um novo modelo
        print("[INFO] Modelo não encontrado. Criando um novo modelo.")
    return modelo

def main():
    dados = carregar_dados_csv()
    entradas, saidas = preparar_dados_treinamento(dados)
    print("[INFO] Dados preparados para treinamento.")
    print(f"[INFO] Total de exemplos: {len(entradas)}")

    # Carrega o modelo salvo, se existir, ou cria um novo modelo
    modelo = carregar_modelo_salvo('modelo_ajustes.keras')
    
    # Continua o treinamento com os dados carregados
    treinar_modelo_basico(modelo, entradas, saidas, dados)

    # Salva o modelo após o treinamento
    modelo.save('modelo_ajustes.keras')
    print("[INFO] Modelo salvo como 'modelo_ajustes.keras'.")

if __name__ == "__main__":
    main()
