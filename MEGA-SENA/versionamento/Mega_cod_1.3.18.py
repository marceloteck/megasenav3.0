import os
import numpy as np
import tensorflow as tf

# Função para carregar os dados do CSV
def carregar_dados_csv(caminho_arquivo):
    dados = np.genfromtxt(caminho_arquivo, delimiter=';', skip_header=1)
    sequencias = dados[:, 1:7]  # Seleciona as colunas N1 a N6
    return sequencias

# Função de cálculo LT
def calcular_LT(sequencia_atual, sequencia_referencia):
    diferencas = sequencia_atual - sequencia_referencia
    ajuste_LT = (diferencas + 9) % 60 - 9  # Ajuste dentro do intervalo [-9, 9]
    return ajuste_LT

# Função de cálculo reverso
def calcular_reverso(ajustes_LT, sequencia_atual):
    ajustes_reverso = []
    for ajuste, numero in zip(ajustes_LT, sequencia_atual):
        novo_numero = numero - ajuste
        if novo_numero <= 0:
            novo_numero += 60
        ajustes_reverso.append(novo_numero)
    return np.array(ajustes_reverso)

# Função para calcular a diferença entre o ajuste esperado e o ajuste atual
def calcular_diferencas_ajustes(sequencia_atual, sequencia_referencia, ajustes_esperados):
    # Calcula os ajustes LT baseados na sequência atual e na sequência de referência
    ajustes_LT = calcular_LT(sequencia_atual, sequencia_referencia)
    ajustados = ajustes_LT + ajustes_esperados  # Ajuste combinado com os ajustes esperados
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
    # Prever os ajustes para a próxima sequência com base no modelo treinado
    previsao_ajustes = modelo(np.expand_dims(ultima_sequencia, axis=0))
    previsao_ajustes = previsao_ajustes.numpy().flatten()

    # Calcular o cálculo LT para prever os próximos números
    ajuste_LT = calcular_LT(ultima_sequencia, ultima_sequencia)  # LT considerando a sequência atual
    proxima_sequencia = ultima_sequencia + ajuste_LT + previsao_ajustes  # LT + Ajustes previstos pela rede

    return proxima_sequencia


# Caminho para o arquivo CSV
caminho_arquivo = 'MEGA-SENA/dados_megasena/Mega_Sena.1.0.csv'

# Carrega os dados
sequencias = carregar_dados_csv(caminho_arquivo)

# Ajustes esperados (baseado no cálculo LT e reverso)
ajustes_esperados = np.zeros_like(sequencias)

# Calcular os ajustes esperados para cada sequência
for i in range(1, len(sequencias)):
    ajustes_esperados[i] = calcular_LT(sequencias[i], sequencias[i-1])

# Treinando até que todos os números sejam acertados
modelo_treinado = treinar_atualizar_ate_acerto(modelo, sequencias, ajustes_esperados, epocas=10)

# Após o treinamento, prever os próximos números
ultima_sequencia = sequencias[-1]
proxima_sequencia = prever_proximos_numeros(modelo_treinado, ultima_sequencia)

print(f"[INFO] A próxima sequência prevista pela rede neural é: {proxima_sequencia}")
