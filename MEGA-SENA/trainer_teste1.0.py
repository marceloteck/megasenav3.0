# Imports e configurações iniciais
import os
# Configurações do ambiente
os.system('title Mega Sena Trainer v1.5.1')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import random
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (Dense, LSTM, Conv1D, MaxPooling1D, Dropout, 
                                     Lambda, Input, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
from tensorflow.keras.optimizers import SGD



# Sementes para reproducibilidade
seed_value = 42
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

# Habilitação da desserialização insegura
keras.config.enable_unsafe_deserialization()

# Variáveis globais e configurações
new_trainer = "noTrainer"  # Opções: "yesTrainer" | "noTrainer"
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_path = 'MEGA-SENA/dados_megasena/'
X_filename = 'MEGA-SENA/BaseDados/prepared_data_X.npy'
y_filename = 'MEGA-SENA/BaseDados/prepared_data_y.npy'
processed_files_filename = 'MEGA-SENA/processed_files.txt'
model_path = ('MEGA-SENA/Trainner/megasena_model_training.keras' if new_trainer == "noTrainer" 
              else f'MEGA-SENA/Trainner/megasena_model_training.{current_time}.keras')

# Funções utilitárias e de processamento de dados
def load_dados(folder_path):
    all_data = pd.concat(
        (pd.read_csv(os.path.join(folder_path, file), sep=';')
         .assign(Data=lambda df: pd.to_datetime(df['Data'], format='%d/%m/%Y'))
         for file in os.listdir(folder_path) if file.endswith('.csv')),
        ignore_index=True
    ).sort_values('Data')
    return all_data

def load_and_preprocess_data(folder_path, processed_files_filename, X_filename):
    processed_files = set()
    if os.path.exists(processed_files_filename):
        with open(processed_files_filename, 'r') as f:
            processed_files = set(f.read().splitlines())

    new_files = [file for file in os.listdir(folder_path) 
                 if file.endswith('.csv') and file not in processed_files or not os.path.exists(X_filename)]
    all_data = [pd.read_csv(os.path.join(folder_path, file), sep=';').assign(Data=lambda df: pd.to_datetime(df['Data'], format='%d/%m/%Y')) 
                for file in new_files]

    if all_data:
        all_data = pd.concat(all_data, ignore_index=True).sort_values('Data')

    with open(processed_files_filename, 'a') as f:
        for file in new_files:
            f.write(file + '\n')

    return all_data

def prepare_data(data, X_filename, y_filename, force_prepare=False):
    if isinstance(data, list):
        data = pd.DataFrame(data)
        
    if not force_prepare and os.path.exists(X_filename) and os.path.exists(y_filename):
        print("Carregando dados preparados dos arquivos...\n")
        X_existing = np.load(X_filename, allow_pickle=True)
        y_existing = np.load(y_filename, allow_pickle=True)
    else:
        print("Preparando os dados...")
        X_existing, y_existing = np.array([]), np.array([])

    if not data.empty:
        print("Adicionando novos dados...")
        X_new = np.array([data.iloc[i - 12:i, 1:].values.flatten() for i in tqdm(range(12, len(data)), desc="Preparando dados", unit="iteração")])
        y_new = np.array([data.iloc[i, 1:].values for i in range(12, len(data))])

        if X_existing.size > 0:
            X = np.concatenate((X_existing, X_new))
            y = np.concatenate((y_existing, y_new))
        else:
            X, y = X_new, y_new

        print("Salvando os dados preparados em arquivos...")
        np.save(X_filename, X, allow_pickle=True)
        np.save(y_filename, y, allow_pickle=True)
    else:
        print("Nenhum dado novo para adicionar.")
        X, y = X_existing, y_existing

    return X, y

# Funções de treinamento e avaliação
def build_model(hp):
    model = Sequential([
        Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
        Conv1D(filters=hp.Int('filters_1', 32, 256, 32), kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        Conv1D(filters=hp.Int('filters_2', 32, 256, 32), kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        LSTM(units=hp.Int('lstm_units', 50, 400, 50), return_sequences=False),
        Dropout(hp.Float('dropout', 0.1, 0.5, 0.1)),
        Dense(units=hp.Int('dense_units', 32, 128, 32), activation='relu'),
        Dense(6),
        Lambda(lambda x: 1 + 59 * x)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                  loss='mean_squared_error')
    return model

def tune_hyperparameters(X_train, y_train):
    tuner = kt.Hyperband(build_model, objective='val_loss', max_epochs=50, factor=3, directory='Memory', project_name='Memory_megasena')
    tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
    return tuner, tuner.get_best_hyperparameters(num_trials=1)[0]

def evaluate_model(X_test_scaled, y_test_scaled, best_model):
    y_pred_scaled = best_model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_rescaled = scaler_y.inverse_transform(y_test_scaled)

    mse = mean_squared_error(y_test_rescaled, y_pred)
    mae = mean_absolute_error(y_test_rescaled, y_pred)
    print(f"Erro Médio Quadrado (MSE): {mse}")
    print(f"Erro Absoluto Médio (MAE): {mae}")

def gerar_dados_sinteticos(dados_reais, num_amostras=1000):
    # Contar a frequência de cada número nos dados reais, excluindo o zero
    todos_numeros = dados_reais.iloc[:, 1:].values.flatten()
    todos_numeros = [num for num in todos_numeros if num != 0]
    frequencia = Counter(todos_numeros)
    total_numeros = sum(frequencia.values())
    
    # Converter as frequências em probabilidades
    probabilidade = {num: freq / total_numeros for num, freq in frequencia.items()}
    
    # Obter as datas já existentes para evitar duplicação
    datas_existentes = set(dados_reais['Data'].dt.strftime('%d/%m/%Y'))
    
    # Gerar dados sintéticos
    dados_sinteticos = []
    for i in range(num_amostras):
        tentativas = 0
        limite_tentativas = 100  # Define um limite para evitar loop infinito
        
        while tentativas < limite_tentativas:
            # Selecionar 6 números baseados nas probabilidades
            numeros = random.choices(
                population=list(probabilidade.keys()), 
                weights=list(probabilidade.values()), 
                k=6
            )
            
            # Verificar se os números estão dentro de uma soma realista
            if 100 <= sum(numeros) <= 300:
                numeros.sort()  # Ordenar os números em ordem crescente
                
                # Gerar uma data aleatória
                dia = random.randint(1, 28)
                mes = random.randint(1, 12)
                ano = random.randint(1996, 2023)
                data_ficticia = f"{dia:02d}/{mes:02d}/{ano}"
                
                if data_ficticia not in datas_existentes:
                    break  # Encontramos uma data única
            tentativas += 1
        
        if tentativas < limite_tentativas:
            dados_sinteticos.append([data_ficticia] + numeros)
            datas_existentes.add(data_ficticia)  # Adicionar a nova data ao conjunto para evitar duplicações
    
    # Converter para DataFrame para facilitar a manipulação posterior
    df_sinteticos = pd.DataFrame(dados_sinteticos, columns=["Data"] + [f"Numero{i+1}" for i in range(6)])
    return df_sinteticos


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
        Dropout(0.3),
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

    entradas_normalizadas = entradas / 60.0

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
    

# Execução principal
if __name__ == "__main__":
    print("Inicializando programa \n")
   

    all_data = load_and_preprocess_data(folder_path, processed_files_filename, X_filename)
    dados = load_dados(folder_path)
    dados_sinteticos = gerar_dados_sinteticos(dados)

    if isinstance(all_data, list) and all_data:
        all_data = pd.concat(all_data, ignore_index=True)
    else:
        all_data = pd.DataFrame()

    if all_data.empty:
        combined_data = dados_sinteticos
    else:
        combined_data = pd.concat([all_data, dados_sinteticos], ignore_index=True)

    combined_data_list = combined_data.values.tolist()

    X, y = prepare_data(combined_data_list, X_filename, y_filename)
    

    print("Iniciando treinamento...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = MinMaxScaler(feature_range=(1, 60))
    scaler_y = MinMaxScaler(feature_range=(1, 60))
    X_train_scaled = scaler_X.fit_transform(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_scaled = scaler_X.transform(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    if os.path.exists(model_path):
        print("\nCarregando modelo salvo...")
        best_model = load_model(model_path)
    else:
        print("\nTreinando o melhor modelo...")
        tuner, best_hps = tune_hyperparameters(X_train_scaled, y_train_scaled)
        best_model = tuner.hypermodel.build(best_hps)
        best_model.fit(X_train_scaled, y_train_scaled, epochs=1, validation_split=0.2, batch_size=16, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

    print("\nContinuando o treinamento com os novos dados...")
    best_model.fit(X_train_scaled, y_train_scaled, epochs=1, validation_split=0.2, batch_size=16, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

    evaluate_model(X_test_scaled, y_test_scaled, best_model)
    best_model.save(model_path)
    print("Modelo salvo.")

    print("\nPrevisão dos próximos números...")
    last_five_draws = dados.iloc[-12:, 1:].values.flatten()
    last_five_scaled = scaler_X.transform(last_five_draws.reshape(1, -1)).reshape(1, 72, 1)
    predicted_scaled = best_model.predict(last_five_scaled)
    predicted_numbers = np.clip(scaler_y.inverse_transform(predicted_scaled).astype(int), 1, 60)
    print("\nNúmeros previstos para o próximo sorteio:", predicted_numbers[0])

    # Continuação do código de previsão...
     
    main()
 



  