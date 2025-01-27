import os
import numpy as np
import tensorflow as tf

# Função para carregar os dados do CSV
def carregar_dados_csv(caminho_arquivo):
    dados = np.genfromtxt(caminho_arquivo, delimiter=';', skip_header=1)
    sequencias = dados[:, 1:7]  # Seleciona as colunas N1 a N6
    return sequencias

# Função para calcular a diferença entre o ajuste esperado e o ajuste atual
def calcular_diferencas_ajustes(sequencia_atual, sequencia_referencia, ajustes_esperados):
    diferencas = sequencia_atual - sequencia_referencia
    ajustados = diferencas + ajustes_esperados
    return ajustados

# Função de perda personalizada
def perda_ajustes_personalizada(y_true, y_pred):
    erro_ajustes = tf.abs(y_true - y_pred)
    penalty = tf.reduce_sum(erro_ajustes)
    regularizacao = tf.reduce_sum(tf.square(y_pred))
    return penalty + 0.01 * regularizacao

# Arquitetura do modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),  # Entrada de 6 números
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(6)  # Saída de 6 ajustes
])

modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=perda_ajustes_personalizada)

# Função de treinamento
def treinar_modelo(modelo, sequencias, ajustes_esperados, epocas=10):
    for epoca in range(epocas):
        for sequencia_atual, sequencia_referencia, ajuste_esperado in zip(sequencias[:-1], sequencias[1:], ajustes_esperados[1:]):
            ajustes_calculados = calcular_diferencas_ajustes(sequencia_atual, sequencia_referencia, ajuste_esperado)

            with tf.GradientTape() as tape:
                previsao = modelo(np.expand_dims(sequencia_atual, axis=0))  # Adiciona uma dimensão extra para o batch
                loss_value = perda_ajustes_personalizada(ajustes_calculados, previsao)

            gradients = tape.gradient(loss_value, modelo.trainable_variables)
            modelo.optimizer.apply_gradients(zip(gradients, modelo.trainable_variables))

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
                ajustes_calculados = calcular_diferencas_ajustes(sequencia_atual, sequencia_referencia, ajuste_esperado)

                with tf.GradientTape() as tape:
                    previsao = modelo(np.expand_dims(sequencia_atual, axis=0))
                    loss_value = perda_ajustes_personalizada(ajustes_calculados, previsao)
                
                gradients = tape.gradient(loss_value, modelo.trainable_variables)
                modelo.optimizer.apply_gradients(zip(gradients, modelo.trainable_variables))

                acertos = calcular_acertos(previsao.numpy(), ajustes_esperados)
                print(f"[INFO] Época {epoca+1}, Acertos: {acertos} de 6")
                print(f"[INFO] Perda para essa iteração: {loss_value.numpy()}")

                if acertos == 6:
                    print("[INFO] Rede Neural acertou todos os números!")
                    return modelo

        print("\n\n[INFO] Rede neural não acertou todos os números, realizando novo treinamento...")

# Função para prever os próximos números
def prever_proximos_numeros(modelo, ultima_sequencia):
    # Prever os ajustes para a próxima sequência
    previsao_ajustes = modelo(np.expand_dims(ultima_sequencia, axis=0))
    previsao_ajustes = previsao_ajustes.numpy().flatten()

    # Calcular os próximos números com base nos ajustes previstos
    proxima_sequencia = ultima_sequencia + previsao_ajustes
    return proxima_sequencia

# Caminho para o arquivo CSV
caminho_arquivo = 'MEGA-SENA/dados_megasena/Mega_Sena.1.0.csv'

# Carrega os dados
sequencias = carregar_dados_csv(caminho_arquivo)

# Ajustes esperados (para cada sequência, você deve calcular o ajuste desejado)
ajustes_esperados = np.random.randint(-9, 10, size=(len(sequencias), 6))

# Treinando até que todos os números sejam acertados
modelo_treinado = treinar_atualizar_ate_acerto(modelo, sequencias, ajustes_esperados, epocas=10)

# Após o treinamento, prever os próximos números
ultima_sequencia = sequencias[-1]
proxima_sequencia = prever_proximos_numeros(modelo_treinado, ultima_sequencia)

print(f"[INFO] A próxima sequência prevista pela rede neural é: {proxima_sequencia}")
