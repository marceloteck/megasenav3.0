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

def add_l2_regularization(original_model, l2_value):
    # Create a list to store the modified layers
    modified_layers = []

    for layer in original_model.layers:
        if isinstance(layer, (Dense, Conv1D, LSTM)):
            # Recreate the layer with L2 regularization
            config = layer.get_config()
            config['kernel_regularizer'] = l2(l2_value)
            new_layer = layer.__class__.from_config(config)
            modified_layers.append(new_layer)
        else:
            # Add layers that don't need modification directly
            modified_layers.append(layer)

    # Build the new model with the modified layers
    new_model = Sequential(modified_layers)
    return new_model


# Carregar o modelo já treinado
model_path = 'megasena_model_automatic_training.2.0.keras'

# Caminho da pasta onde os arquivos CSV estão armazenados
folder_path = 'dados_megasena/'

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

print("\n")
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
        y.append(data.iloc[i, 1:].values)

    X, y = np.array(X), np.array(y)
    return X, y

X, y = prepare_data(all_data)


print("\n")

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados diretamente no intervalo de 1 a 25
scaler_X = MinMaxScaler(feature_range=(1, 60))
scaler_y = MinMaxScaler(feature_range=(1, 60))
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
        Conv1D(200, 3, activation='relu'), # 100 -> 200
        MaxPooling1D(pool_size=2),
        Conv1D(308, 3, activation='relu'), # 128 -> 308
        MaxPooling1D(pool_size=2),
        LSTM(300, return_sequences=False, kernel_regularizer=l2(0.001)), # 0.001  -> 300
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)), # 0.001 DENSE: 128
        Dropout(0.5),
        Dense(6),  # 15 números sorteados na Lotofácil
        Lambda(lambda x: 1 + 59 * x)  # Escalando para o intervalo 1-25 -> 1 + 24
    ])

# Ajuste de taxa de aprendizado com ReduceLROnPlateau
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=0.001) # 0.0001

# Callback EarlyStopping para evitar overfitting (parada prematura)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)


if os.path.exists(model_path):
  # Clonar o modelo original para manter a arquitetura
  new_model = add_l2_regularization(model, l2_value=0.01)
  # Compilação do modelo
  new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='mean_absolute_error')

  # Copiar os pesos do modelo original para o novo modelo, camada por camada
  for i, layer in enumerate(new_model.layers):
      if layer.get_weights():  # Check if layer has weights
          layer.set_weights(model.layers[i].get_weights())

  # Treinamento com EarlyStopping e LearningRateScheduler
  print("Iniciando treinamento... melhorado")
  history = new_model.fit(
      X_train_scaled,
      y_train_scaled,
      epochs=3,  # Use mais épocas para melhor performance 400
      batch_size=16,
      validation_split=0.2,
      callbacks=[early_stopping, lr_scheduler]
  )
else:
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mean_absolute_error') # 0.0005 -> learning_rate=0.0001

  # Treinamento com EarlyStopping e LearningRateScheduler
  print("Iniciando treinamento...")
  history = model.fit(
      X_train_scaled,
      y_train_scaled,
      epochs=2,  # Use mais épocas para melhor performance 400
      batch_size=32,
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

# Obter os últimos cinco sorteios
last_fifteen_draws = data.iloc[-5:, 1:].values.flatten()

# Garantir que o tamanho é exatamente 75 (5 sorteios * 15 números)
if last_fifteen_draws.size > 30:
    last_five_draws = last_fifteen_draws [:30]  # Cortar o excesso
elif last_fifteen_draws.size < 30:
    raise ValueError(f"Tamanho insuficiente de dados para redimensionar: necessário 225, mas foi encontrado {last_fifteen_draws.size}.")

# Normalizando e ajustando a forma para (1, 225, 1)
last_five_scaled = scaler_X.transform(last_fifteen_draws.reshape(1, -1))
last_five_scaled = last_five_scaled.reshape(1, 30, 1)

# Predição dos próximos números
predicted_scaled = model.predict(last_five_scaled)

# Desescalonando as previsões e arredondando para os números mais próximos
predicted_numbers = scaler_y.inverse_transform(predicted_scaled)
predicted_numbers = np.round(predicted_numbers).astype(int)

# Garantir que os números estão no intervalo de 1 a 25
predicted_numbers = np.clip(predicted_numbers, 1, 60)


print("\n")
print(f"Números previstos para o próximo sorteio: {predicted_numbers[0]}")

print("\n")
print("\n")

# Visualizar a evolução do treinamento
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Loss de Treino')
plt.plot(history.history['val_loss'], label='Loss de Validação')
plt.title('Evolução do Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
