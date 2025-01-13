import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, Lambda, Input
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm


# Habilitar a desserialização insegura
keras.config.enable_unsafe_deserialization()
# Carregar o modelo já treinado
model_path = 'lotofacil_model_automatic_training.2.0.keras'

# Caminho da pasta onde os arquivos CSV estão armazenados
folder_path = 'dados_lotofacil/'

# Inicializar um DataFrame vazio para armazenar todos os dados combinados
all_data = pd.DataFrame()

print("Arquivos para análise:")
# Percorrer todos os arquivos na pasta
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):  # Verificar se o arquivo é um CSV
        file_path = os.path.join(folder_path, filename)

        print(filename)

        # Carregar o arquivo CSV
        data = pd.read_csv(file_path, sep=';')

        # Converter a coluna 'Data' para o formato datetime
        data['Data'] = pd.to_datetime(data['Data'], format='%d/%m/%Y')

        # Concatenar os dados carregados com o DataFrame existente
        all_data = pd.concat([all_data, data], ignore_index=True)

# Ordenar os dados pela coluna 'Data'
all_data = all_data.sort_values('Data')


# Função para preparar os dados para a entrada da rede neural
print("Iniciando preparação dos dados")

import numpy as np

def prepare_data(data):
    X, y = [], []
    total = len(data) - 5  # Total de iterações

    # Usando tqdm para adicionar a barra de progresso
    for i in tqdm(range(5, len(data)), desc="Preparando dados", unit="iteração"):
        X.append(data.iloc[i - 5:i, 1:].values.flatten())
         # Selecionando apenas as primeiras 15 colunas para y
        y.append(data.iloc[i, 1:].values)

    X, y = np.array(X), np.array(y)
    return X, y

X, y = prepare_data(all_data)


print("\n")

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados diretamente no intervalo de 1 a 25
scaler_X = MinMaxScaler(feature_range=(1, 25))
scaler_y = MinMaxScaler(feature_range=(1, 25))
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Ajustar a forma dos dados para a CNN e LSTM
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

if os.path.exists(model_path):
    print("Carregando o modelo treinado previamente...")
    model = load_model(model_path)
else:
    print("Modelo não encontrado, iniciando treinamento...")
    # Defina o seu modelo e continue o treinamento
    # Definir o modelo de rede neural com CNN + LSTM
    model = Sequential([
        Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),  # Usando a camada Input
        Conv1D(200, 3, activation='relu'), # 100
        MaxPooling1D(pool_size=2),
        Conv1D(308, 3, activation='relu'), # 128
        MaxPooling1D(pool_size=2),
        LSTM(300, return_sequences=False, kernel_regularizer=l2(0.001)), # 0.001
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)), # 0.001 DENSE: 128
        Dropout(0.5),
        Dense(15),  # 15 números sorteados na Lotofácil
        Lambda(lambda x: 1 + 15 * x)  # Escalando para o intervalo 1-25
    ])

# Ajuste de taxa de aprendizado com ReduceLROnPlateau
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=0.001) # 0.0001

# Callback EarlyStopping para evitar overfitting (parada prematura)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# Compilação do modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mean_absolute_error') # 0.0005

# Treinamento com EarlyStopping e LearningRateScheduler
print("Iniciando treinamento...")
history = model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=5,  # Use mais épocas para melhor performance 400
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stopping, lr_scheduler]
)

# Avaliação no conjunto de teste
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_rescaled = scaler_y.inverse_transform(y_test_scaled)

print("Treinamento Concluído.")
print("\n")

# Avaliar as métricas
mse = mean_squared_error(y_test_rescaled, y_pred)
mae = mean_absolute_error(y_test_rescaled, y_pred)
print(f"Erro Médio Quadrado (MSE): {mse}")
print(f"Erro Absoluto Médio (MAE): {mae}")

# Salvar o modelo treinado
model.save(model_path)
print("Modelo salvo.")
"""
##################################
# Previsão dos próximos números
last_fifteen_draws = data.iloc[-5:, 1:].values

# Normalizando e ajustando a forma para (1, 255)
last_fifteen_draws = last_fifteen_draws.reshape(1, -1)
last_five_scaled = last_five_scaled.reshape(1, 75, 1)

# Predição de várias amostras e coleta dos números mais frequentes
n_predictions = 1000
all_predicted_numbers = []

#for _ in range(n_predictions):
# Predição dos próximos números apenas uma vez
predicted_scaled = model.predict(last_five_scaled)
predicted_numbers = scaler_y.inverse_transform(predicted_scaled)
predicted_numbers = np.round(predicted_numbers).astype(int)
predicted_numbers = np.clip(predicted_numbers, 1, 25)

# Contar as frequências dos números previstos
number_counts = pd.Series(all_predicted_numbers).value_counts()
most_frequent_numbers = number_counts.head(30).index.tolist()  # Selecionar os 30 números mais frequentes

# Gerar 7 jogadas distintas a partir dos números mais frequentes
from itertools import combinations
from collections import Counter

# Contar a frequência de cada número no conjunto de treinamento
number_counts = Counter(y_train.flatten())

# Selecionar os 30 números mais frequentes
top_30_numbers = [num for num, _ in number_counts.most_common(30)]


# Formar todas as combinações possíveis de 15 números a partir dos 30 mais frequentes
all_combinations = list(combinations(top_30_numbers, 15))

# Garantir que há pelo menos 5 combinações para escolher
if len(all_combinations) < 5:
    jogadas = all_combinations  # Seleciona todas as combinações disponíveis
else:
    # Selecionar 8 combinações aleatórias
    eight_combinations = np.random.choice(range(len(all_combinations)), 5, replace=False)
    jogadas = [all_combinations[i] for i in eight_combinations]

"""
# Obter os últimos cinco sorteios
last_fifteen_draws = data.iloc[-5:, 1:].values.flatten()

# Garantir que o tamanho é exatamente 75 (5 sorteios * 15 números)
if last_fifteen_draws.size > 75:
    last_five_draws = last_fifteen_draws [:75]  # Cortar o excesso
elif last_fifteen_draws.size < 75:
    raise ValueError(f"Tamanho insuficiente de dados para redimensionar: necessário 225, mas foi encontrado {last_fifteen_draws.size}.")

# Normalizando e ajustando a forma para (1, 225, 1)
last_five_scaled = scaler_X.transform(last_fifteen_draws.reshape(1, -1))
last_five_scaled = last_five_scaled.reshape(1, 75, 1)

# Predição dos próximos números
predicted_scaled = model.predict(last_five_scaled)

# Desescalonando as previsões e arredondando para os números mais próximos
predicted_numbers = scaler_y.inverse_transform(predicted_scaled)
predicted_numbers = np.round(predicted_numbers).astype(int)

# Garantir que os números estão no intervalo de 1 a 25
predicted_numbers = np.clip(predicted_numbers, 1, 25)
"""
print("\n")
# Exibir as 8 jogadas formadas
for i, jogada in enumerate(jogadas, start=1):
    print(f"Jogada {i}: {sorted(jogada)}")
"""
print("\n")
print(f"Números previstos para o próximo sorteio: {predicted_numbers[0]}")

