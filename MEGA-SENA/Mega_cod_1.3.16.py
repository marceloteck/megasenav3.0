import os

# Configuração do ambiente TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf


# Função para carregar os dados do CSV
def carregar_dados_csv(caminho_arquivo):
    # Carregar os dados do arquivo CSV
    dados = np.genfromtxt(caminho_arquivo, delimiter=';', skip_header=1)
    sequencias = dados[:, 1:7]  # Seleciona as colunas N1 a N6
    return sequencias

# Função para calcular a diferença entre o ajuste esperado e o ajuste atual
def calcular_diferencas_ajustes(sequencia_atual, sequencia_referencia, ajustes_esperados):
    # Calcula as diferenças entre a sequência atual e a referência
    diferencas = sequencia_atual - sequencia_referencia
    ajustados = diferencas + ajustes_esperados
    return ajustados

# Função de perda personalizada
def perda_ajustes_personalizada(y_true, y_pred):
    # Penaliza a diferença entre o ajuste real e o ajustado pela rede
    erro_ajustes = tf.abs(y_true - y_pred)
    
    # Penaliza mais fortemente grandes diferenças
    penalty = tf.reduce_sum(erro_ajustes)
    
    # Adiciona uma regularização para evitar overfitting
    regularizacao = tf.reduce_sum(tf.square(y_pred))
    
    return penalty + 0.01 * regularizacao

# Ajuste da taxa de aprendizado para otimização
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Arquitetura do modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),  # Entrada de 6 números
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(6)  # Saída de 6 ajustes
])

modelo.compile(optimizer=optimizer, loss=perda_ajustes_personalizada)

# Função de treinamento
def treinar_modelo(modelo, sequencias, ajustes_esperados, epocas=10):
    for epoca in range(epocas):
        for sequencia_atual, sequencia_referencia, ajuste_esperado in zip(sequencias[:-1], sequencias[1:], ajustes_esperados[1:]):
            # Calcula a diferença dos ajustes esperados
            ajustes_calculados = calcular_diferencas_ajustes(sequencia_atual, sequencia_referencia, ajuste_esperado)

            with tf.GradientTape() as tape:
                # Faz a previsão dos ajustes pela rede
                previsao = modelo(np.expand_dims(sequencia_atual, axis=0))  # Adiciona uma dimensão extra para o batch

                
                # Calcula a perda com base nos ajustes esperados
                loss_value = perda_ajustes_personalizada(ajustes_calculados, previsao)
            
            # Calcula os gradientes e aplica a atualização dos pesos
            gradients = tape.gradient(loss_value, modelo.trainable_variables)
            optimizer.apply_gradients(zip(gradients, modelo.trainable_variables))
        
        print(f"[INFO] Época {epoca+1} completa com perda: {loss_value.numpy()}")

# Função para calcular os acertos da rede
def calcular_acertos(previsao, ajuste_esperado):
    acertos = np.sum(np.isclose(previsao, ajuste_esperado, atol=1))
    return acertos

# Função de treinamento contínuo até acertos completos
def treinar_atualizar_ate_acerto(modelo, sequencias, ajustes_esperados, epocas=10):
    while True:
        for epoca in range(epocas):
            for sequencia_atual, sequencia_referencia, ajuste_esperado in zip(sequencias[:-1], sequencias[1:], ajustes_esperados[1:]):
                # Calcula a diferença dos ajustes esperados
                ajustes_calculados = calcular_diferencas_ajustes(sequencia_atual, sequencia_referencia, ajuste_esperado)

                with tf.GradientTape() as tape:
                    # Adiciona uma dimensão extra para o batch
                    previsao = modelo(np.expand_dims(sequencia_atual, axis=0))
                    
                    # Calcula a perda com base nos ajustes esperados
                    loss_value = perda_ajustes_personalizada(ajustes_calculados, previsao)
                
                # Calcula os gradientes e aplica a atualização dos pesos
                gradients = tape.gradient(loss_value, modelo.trainable_variables)
                optimizer.apply_gradients(zip(gradients, modelo.trainable_variables))

                # Calcula os acertos
                acertos = calcular_acertos(previsao.numpy(), ajustes_esperados)
                print(f"[INFO] Época {epoca+1}, Acertos: {acertos} de 6")
                print(f"[INFO] Perda para essa iteração: {loss_value.numpy()}")

                # Se a rede acertar todos os números, interrompe o treinamento
                if acertos == 6:
                    print("[INFO] Rede Neural acertou todos os números!")
                    return modelo  # Retorna o modelo treinado

        print("[INFO] Rede neural não acertou todos os números, realizando novo treinamento...")


# Caminho para o arquivo CSV
caminho_arquivo = 'MEGA-SENA/dados_megasena/Mega_Sena.1.0.csv'

# Carrega os dados
sequencias = carregar_dados_csv(caminho_arquivo)

# Ajustes esperados (para cada sequência, você deve calcular o ajuste desejado)
# Este é apenas um exemplo, você deve calcular os ajustes conforme a sua lógica
# Aqui estamos usando ajustes fictícios como exemplo:
ajustes_esperados = np.random.randint(-9, 10, size=(len(sequencias), 6))

# Treinando até que todos os números sejam acertados
modelo_treinado = treinar_atualizar_ate_acerto(modelo, sequencias, ajustes_esperados, epocas=10)
