import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

# Função para adicionar novas características ao dataset
def add_features(df):
    # Frequência dos números
    frequency = {i: 0 for i in range(1, 61)}
    for row in df.itertuples(index=False):
        for number in row:
            frequency[number] += 1

    # Adiciona colunas de freqüência
    for i in range(1, 61):
        df[f'freq_{i}'] = frequency[i]

    # Adiciona colunas de média móvel e desvio padrão
    df['mean'] = df.mean(axis=1)
    df['std'] = df.std(axis=1)

    return df

# Exemplo de dados simulados
data = {
    'number_1': [5, 16, 22, 28, 33, 42],
    'number_2': [8, 19, 23, 29, 36, 45],
    'number_3': [10, 20, 24, 30, 38, 48],
    'number_4': [15, 25, 26, 31, 40, 50],
    'number_5': [18, 28, 29, 32, 43, 55],
    'number_6': [21, 30, 35, 34, 47, 58]
}

df = pd.DataFrame(data)

# Adiciona novas características
augmented_df = add_features(df)

# Separa as entradas (X) e saídas (y)
X = augmented_df.iloc[:, :-6]  # Todas as colunas exceto as últimas 6
y = augmented_df.iloc[:, -6:]  # Últimas 6 colunas

# Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Criação do modelo
model = Sequential([
    Dense(128, input_dim=X_scaled.shape[1], activation='relu'),
    Dense(64, activation='relu'),
    Dense(6)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Treinamento do modelo
model.fit(X_scaled, y, epochs=500, validation_split=0.2, batch_size=16, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

print("Modelo treinado com sucesso!")
