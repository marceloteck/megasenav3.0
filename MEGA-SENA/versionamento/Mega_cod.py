import os

# Configurações do ambiente
os.system('title Mega Sena Trainer v1.6')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


# Função para carregar e pré-processar os dados
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath, delimiter=';')
    """ print(data.head())  # Verifica a estrutura dos dados
    print(data.columns)  # Verifica as colunas disponíveis
    print(f"Total rows in CSV: {len(data)}")  # Verifica o número total de linhas no CSV
    """
    sequences = []
    labels = []

    for i in range(len(data) - 36):
        sequence = data.iloc[i:i+36, 1:].values  # Assumindo que as colunas N1 a N6 estão do índice 1 a 6
        label = data.iloc[i+36, 1:].values
        label_encoded = to_categorical(label - 1, num_classes=60)

        sequences.append(sequence)
        labels.append(label_encoded)

    sequences = np.array(sequences)
    labels = np.array(labels)
    
    # Verificando as últimas linhas não usadas
    unused_data = data.iloc[-36:]
    """print("Últimas linhas não utilizadas nas sequências:")
    print(unused_data)
    
    print(f"Shape of sequences: {sequences.shape}")
    print(f"Shape of labels: {labels.shape}")
    """
    return sequences, labels





def build_rnn_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(1024, return_sequences=False, kernel_regularizer=l2(0.01))(inputs)  # Aumentando para 1024 neurônios
    x = Dropout(0.3)(x)  # Aumentando o dropout
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)  # Aumentando para 512 neurônios
    model = Model(inputs, x)
    return model

def build_dnn_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Dense(512, activation='relu', kernel_regularizer=l2(0.01)),  # Aumentando para 512 neurônios
        Dropout(0.3),  # Aumentando o dropout
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),  # Aumentando para 256 neurônios
        Dense(6 * 60, activation='softmax'),  # Saída ajustada para corresponder à forma (None, 6, 60)
        Reshape((6, 60))  # Reformatando a saída para (None, 6, 60)
    ])
    model.compile(optimizer='adam', loss=CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model


# Função para treinar a DNN
def train_dnn_model(model, data, labels, epochs=100, batch_size=32):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(data, labels, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping])
    return model

# Função para ajuste contínuo do modelo com novos dados
def fine_tune_model(model, new_data, new_labels, epochs=10, batch_size=32):
    model.fit(new_data, new_labels, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return model

# Função para fazer previsões
def make_predictions(model, data):
    predictions = model.predict(data)
    predictions = predictions.reshape(-1, 6, 60)
    predicted_numbers = np.argmax(predictions, axis=2) + 1
    return predicted_numbers

# Função para análise de erros
def error_analysis(predictions, actual):
    differences = np.sum(predictions == actual, axis=1)
    mean_error = np.mean(differences)
    std_error = np.std(differences)
    return {'mean_error': mean_error, 'std_error': std_error}

# Função para reforçar o aprendizado baseado nos erros
def reinforce_learning(model, predictions, actual, data, labels, learning_rate=0.01):
    # Verifique o tamanho dos índices de erro e ajuste se necessário
    error_indices = np.where(np.any(predictions != actual, axis=1))[0]

    if len(error_indices) > 0:
        error_data = data[error_indices]
        error_labels = labels[error_indices]
        
        model.fit(error_data, error_labels, epochs=5, batch_size=16, verbose=0)
        
    return model

# Fluxo de execução
if __name__ == "__main__":
    filepath = 'MEGA-SENA/dados_megasena/Mega_Sena.csv'  # Substitua pelo caminho correto do arquivo de dados
    
    # Testando a função
    load_and_preprocess_data(filepath)

    data, labels = load_and_preprocess_data(filepath)

    # Divisão dos dados para treinamento inicial e validação
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Criando e treinando a RNN/LSTM
    rnn_model = build_rnn_model(input_shape=(36, 6))
    rnn_train_output = rnn_model.predict(train_data)

    # Reshape the RNN output to match DNN input
    rnn_train_output = rnn_train_output.reshape(-1, 1024)  # Altere para 1024, conforme o número de neurônios na LSTM

    # Criando e treinando a DNN com as saídas da RNN/LSTM
    dnn_model = build_dnn_model(input_shape=(1024,))
    dnn_model = train_dnn_model(dnn_model, rnn_train_output, train_labels)

    # Processamento de dados de validação através da RNN
    rnn_val_output = rnn_model.predict(val_data)
    rnn_val_output = rnn_val_output.reshape(-1, 1024)  # Certifique-se de que o tamanho corresponda
    
    # Verifique o tamanho de rnn_val_output e val_labels
    print(f"Tamanho de rnn_val_output: {rnn_val_output.shape}, Tamanho de val_labels: {val_labels.shape}")


    """
    # Avaliação do modelo
    predictions = make_predictions(dnn_model, rnn_val_output)
    actual_numbers = np.argmax(val_labels, axis=2) + 1
    """


    # Supondo que `predictions` e `actual_numbers` sejam arrays numpy
    predictions = np.random.randint(0, 60, (278, 6))  # Exemplo de previsões
    actual_numbers = np.random.randint(0, 60, (278, 6))  # Exemplo de números reais

    # Verificando as formas antes de comparar
   # print(f"Shape of predictions: {predictions.shape}")
    #print(f"Shape of actual_numbers: {actual_numbers.shape}")

    # Certificando-se de que as formas são compatíveis
    if predictions.shape == actual_numbers.shape:
        error_indices = np.where(np.sum(predictions != actual_numbers, axis=1) > 0)[0]
       # print("Índices com erros:", error_indices)
    else:
        print("As formas de predictions e actual_numbers não são compatíveis.")
    
    #print(f"\n\nPrevisões: {predictions}\n\n")

    # Ordenando as previsões e números reais (caso necessário para comparação)
    predictions_sorted = np.sort(predictions, axis=1)
    actual_numbers_sorted = np.sort(actual_numbers, axis=1)

    # Verificando as formas antes de comparar
    #print(f"Shape of predictions: {predictions_sorted.shape}")
    #print(f"Shape of actual_numbers: {actual_numbers_sorted.shape}")

    # Certificando-se de que as formas são compatíveis
    if predictions_sorted.shape == actual_numbers_sorted.shape:
        error_indices = np.where(np.sum(predictions_sorted != actual_numbers_sorted, axis=1) > 0)[0]
       # print("Índices com erros:", error_indices)
    else:
        print("As formas de predictions e actual_numbers não são compatíveis.")

    # Exibindo as previsões
    print(f"\n\nPrevisões (ordenadas): \n{predictions_sorted}")
    
    # Salvando as previsões ordenadas em um arquivo txt
    with open("previsoes_ordenadas.txt", "w") as file:
        file.write("Previsões ordenadas:\n")
        for i, prediction in enumerate(predictions_sorted):
            file.write(f"Sorteio {i+1}: {prediction}\n")

    print("Previsões salvas no arquivo 'previsoes_ordenadas.txt'")