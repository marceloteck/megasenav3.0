import os

# Configuração do ambiente TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD


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
        Dense(128, input_dim=6, activation='relu'),
        Dropout(0.2),  # 20% de Dropout
        Dense(64, activation='relu'),
        Dense(6, activation='linear')
    ])
    modelo.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='mse', metrics=['accuracy'])
    return modelo
    
    
# Função de treinamento com cálculo LT e comparação com a rede neural
def treinar_modelo_basico(modelo, entradas, saidas, dados):
    acertos_lt_totais = {6: 0, 5: 0, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0}
    acertos_rede_totais = {6: 0, 5: 0, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0}
    
    # Histórico de perda para visualização
    historico_perda = []
    
    # Normalização das entradas
    entradas_normalizadas = entradas / 60.0
    



    # Realizando o treinamento por épocas
    for i in range(len(entradas)):
        L_ATUAL = 6 + i
        L_ESPER = L_ATUAL + 1
        concurso_atual = dados.iloc[L_ATUAL][['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values
        
        L_REFER = L_ATUAL - concurso_atual[0]
                        
        concurso_referencia = dados.iloc[L_REFER][['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values
        resultado_esperado = dados.iloc[L_ESPER][['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values

        # Calcular LT
        previsao_lt = calcular_lt(concurso_atual, concurso_referencia)
        # Calcular Ajuste
        ajustes = calcular_ajuste(previsao_lt, resultado_esperado)



        # Resultado LT + Ajuste Esperado (Resultado LT Plus)
        resultado_lt_plus = [int(previsao_lt[j] + ajustes[j]) for j in range(6)]

        # Verificar os acertos do cálculo LT
        acerto_lt = sum(1 for a, b in zip(resultado_lt_plus, resultado_esperado) if a == b)
        acertos_lt_totais[acerto_lt] += 1

        # Treinamento da rede neural
        historico = modelo.fit(entradas_normalizadas, saidas, batch_size=16, epochs=10, verbose=0)
        historico_perda.append(historico.history['loss'][0])

        # Previsão da rede neural para a entrada atual
        previsao_rede = modelo.predict(np.array([entradas_normalizadas[i]]))  # Predição para a entrada i
        previsao_rede = arredondar_ajustes(previsao_rede[0] * 9.0) # Arredondar a previsão para inteiros

        # Verificar os acertos da rede neural
        acertos_rede = sum(1 for a, b in zip(ajustes, previsao_rede) if a == b)
        acertos_rede_totais[acertos_rede] += 1
        resultado_rd_plus = [int(previsao_lt[j] + previsao_rede[j]) for j in range(6)]        

        # Exibir comparações
        #if i % 10 == 0:  # Exibe a cada 10 iterações
        print("\n")
        print(f"[INFO] Concurso Referência (Linha {L_REFER}): {list(map(int, concurso_referencia))}")
        print(f"[INFO] Concurso Atual (Linha {L_ATUAL}): {list(map(int, concurso_atual))}")
        print(f"[INFO] Resultado Esperado (Linha {L_ESPER}): {list(map(int, resultado_esperado))}")
        print("\n")
        print(f"[INFO] Cálculo LT: {list(map(int, previsao_lt))}")
        print(f"[INFO] Ajustes Esperados: {list(map(int, ajustes))}")
        print(f"[INFO] Resultado LT + AJUSTES: {list(map(int, resultado_lt_plus))}")
        print(f"[INFO] Acertos do Cálculo LT: {acerto_lt} acertos")
        print("\n")
        print(f"[INFO] Previsão do AJUSTE da Rede Neural: {list(map(int, previsao_rede))}")
        print(f"[INFO] Previsão da prx sequencia Rede Neural: {list(map(int, resultado_rd_plus))}")
        print(f"[INFO] Acertos da Rede Neural: {acertos_rede} acertos")
        print("-" * 50)
        print("\n")

    # Relatório Final de Acertos
    print("\n[INFO] Relatório Final de Acertos do Cálculo LT:")
    for acertos, quantidade in acertos_lt_totais.items():
        print(f"{acertos} acertos: {quantidade} vez(es)")

    print("\n[INFO] Relatório Final de Acertos da Rede Neural:")
    for acertos, quantidade in acertos_rede_totais.items():
        print(f"{acertos} acertos: {quantidade} vez(es)")

    # Exibir o histórico de perda para ver como a rede está aprendendo
    print(historico_perda)
    import matplotlib.pyplot as plt
    plt.plot(historico_perda)
    plt.title("Histórico de Perda do Treinamento")
    plt.xlabel("Épocas")
    plt.ylabel("Perda (Loss)")
    plt.show()


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