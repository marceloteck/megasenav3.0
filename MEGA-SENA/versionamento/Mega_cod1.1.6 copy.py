import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input

# Configurações do ambiente
os.system('title Mega Código 2.0')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Função para criar o modelo de rede neural
def criar_modelo():
    modelo = Sequential([
        Input(shape=(18,)),  # 6 diferenças + 12 somas
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(6)  # Saída para os 6 ajustes numéricos
    ])
    modelo.compile(optimizer='adam', loss='mse')
    return modelo

# Função para carregar o modelo existente ou criar um novo
def carregar_ou_criar_modelo():
    if os.path.exists('modelo_megasena.keras'):
        return load_model('modelo_megasena.keras')
    else:
        return criar_modelo()

# Função para carregar e preprocessar os dados
def carregar_dados(filepath):
    df = pd.read_csv(filepath, sep=';')
    df = df[['N1', 'N2', 'N3', 'N4', 'N5', 'N6']]
    return df.values

def gerar_previsao_unica(previsao):
    # Ajusta a previsão para garantir que ela tenha 6 números únicos entre 1 e 60
    previsao_unica = set(previsao)  # Remove duplicatas
    while len(previsao_unica) < 6:
        # Se a previsão tiver menos de 6 números únicos, preenche com números aleatórios
        previsao_unica.add(np.random.randint(1, 61))
    
    return np.array(list(previsao_unica))  # Converte de volta para um array com 6 números únicos

def calcular_previsao(seq_referencia, ajustes):
    melhores_ajustes = []
    ajustes_realizados = []  # Lista para armazenar os ajustes feitos
    for i, ajuste in enumerate(ajustes):
        previsao = seq_referencia + ajuste
        previsao = np.clip(previsao, 1, 60)  # Garantir que os números estejam entre 1 e 60
        diferenca = previsao - seq_referencia
        
        # Soma dos dígitos de cada número da sequência (diferenca e somas)
        somas = [sum(map(int, str(abs(num)))) for num in seq_referencia] + \
                [sum(map(int, str(abs(num)))) for num in previsao]
        
        # Garantir que 'diferenca' e 'somas' sejam arrays 1D
        diferenca = np.array(diferenca)
        somas = np.array(somas)
        
        # Concatenando as diferenças e somas
        entrada = np.concatenate((diferenca, somas))
        
        # Garantir que a entrada tenha 18 elementos (ajuste conforme necessário)
        if entrada.shape[0] > 18:
            entrada = entrada[:18]  # Truncar para 18 elementos
        elif entrada.shape[0] < 18:
            entrada = np.pad(entrada, (0, 18 - entrada.shape[0]), 'constant')  # Preencher com zeros se necessário
        
        melhores_ajustes.append(entrada)

        # Imprimir o ajuste feito para cada número da sequência
        ajustes_realizados.append((seq_referencia, previsao, ajuste))
    
    # Imprimir ajustes realizados
    print("Ajustes realizados para a sequência de referência:", seq_referencia)
    for seq_ref, previsao, ajuste in ajustes_realizados:
        print(f"Ajuste {ajuste}: {seq_ref} -> {previsao}")
    
    return np.array(melhores_ajustes)

    
    # Imprimir ajustes realizados
    print("Ajustes realizados para a sequência de referência:", seq_referencia)
    for num_ref, previsao, ajuste in ajustes_realizados:
        print(f"Ajuste {ajuste}: {num_ref} -> {previsao}")
    
    return np.array(melhores_ajustes)

# Função para realizar o treinamento supervisionado
def treinamento_supervisionado_com_calculos_auto_v2(modelo, sequencias, start_line=15, max_tentativas=3):
    ajustes = np.array([1, -1, 2, -2, 3, -3, 6, -6, 9, -9])

    for i in range(start_line, len(sequencias) - 1):
        seq_atual = sequencias[i]
        seq_proxima = sequencias[i + 1]
        num_linhas_voltar = seq_atual[0]

        if i - num_linhas_voltar < 0:
            print(f"Não há linhas suficientes para calcular para a sequência {seq_atual}. Pulando.")
            continue

        seq_referencia = sequencias[i - num_linhas_voltar]
        entradas_ajustadas = calcular_previsao(seq_referencia, ajustes)

        # Garantir que a entrada tenha o formato correto (1, 18)
        entradas_ajustadas = np.array(entradas_ajustadas)
        
        # Se o número de elementos não for 18, ajustar (truncar ou preencher com zeros)
        if entradas_ajustadas.shape[1] > 18:
            entradas_ajustadas = entradas_ajustadas[:, :18]  # Truncar para 18 elementos
        elif entradas_ajustadas.shape[1] < 18:
            entradas_ajustadas = np.pad(entradas_ajustadas, ((0, 0), (0, 18 - entradas_ajustadas.shape[1])), 'constant')  # Preencher com zeros

        previsoes = modelo.predict(entradas_ajustadas)
        previsao_final = np.round(previsoes).astype(int).flatten()
        previsao_final = np.clip(previsao_final, 1, 60)

        # Garantir que os números sejam únicos
        previsao_final = gerar_previsao_unica(previsao_final)

        print(f"\nSequência atual da planilha (linha {i + 1}): {seq_atual}")
        print(f"Sequência de referência (linha {i + 1 - num_linhas_voltar}): {seq_referencia}")
        print(f"Previsão final feita pelo modelo: {previsao_final}")
        print(f"Sequência real da próxima linha (linha {i + 2}): {seq_proxima}")

        # Comparação automática
        tentativas = 0
        while not np.array_equal(previsao_final, seq_proxima) and tentativas < max_tentativas:
            print(f"Tentativa {tentativas + 1} falhou. Tentando novamente...")
            tentativas += 1
            previsoes = modelo.predict(entradas_ajustadas)
            previsao_final = np.round(previsoes).astype(int).flatten()
            previsao_final = np.clip(previsao_final, 1, 60)
            previsao_final = gerar_previsao_unica(previsao_final)

        # Se a previsão estiver correta ou após as tentativas
        if np.array_equal(previsao_final, seq_proxima):
            print("Previsão correta!")
            modelo.save('modelo_megasena.keras')
        else:
            print(f"Não foi possível acertar a sequência após {max_tentativas} tentativas. Continuando.")

        # Re-treinamento caso o modelo tenha acertado
        if np.array_equal(previsao_final, seq_proxima):
            modelo.fit(entradas_ajustadas, seq_proxima.reshape(1, -1), epochs=10, verbose=0)
            print("Re-treinamento concluído.")
            modelo.save('modelo_megasena.keras')

# Caminho para o arquivo CSV
filepath = 'MEGA-SENA/dados_megasena/Mega_Sena.csv'

# Carregando e preprocessando os dados
sequencias = carregar_dados(filepath)

# Carregando ou criando o modelo
modelo = carregar_ou_criar_modelo()

# Realizando o treinamento supervisionado com cálculos exibidos
treinamento_supervisionado_com_calculos_auto_v2(modelo, sequencias)
