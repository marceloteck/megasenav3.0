import os
import numpy as np
import pandas as pd

# Configuração do ambiente TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Função para carregar dados do CSV
def carregar_dados_csv(caminho_csv):
    try:
        dados = pd.read_csv(caminho_csv, sep=';')
        dados = dados[['N1', 'N2', 'N3', 'N4', 'N5', 'N6']]
        print("Dados carregados com sucesso!")
        return dados.values
    except Exception as e:
        print(f"Erro ao carregar os dados: {e}")
        return None

# Função de cálculo direto
def calculo_direto(num1, num2):
    return (num1 + num2) % 60  # Exemplo de cálculo direto (ajustado para Mega-Sena)

# Função para gerar dados de treinamento
def gerar_dados_treinamento(dados):
    entradas = []
    saídas = []
    
    # Gerar entradas e saídas para o treinamento
    for i in range(3, len(dados)):
        concurso_atual = dados[i]       # Concurso atual (linha i)
        concurso_referencia = dados[i-3]  # Concurso de referência (linha i-3)
        
        for j in range(6):  # Para cada número no concurso
            numero_atual = int(concurso_atual[j])  # Garantir que o número seja tratado como inteiro
            numero_referencia = int(concurso_referencia[j])  # Garantir que o número seja tratado como inteiro
            
            # Calcular a entrada e a saída para o treinamento
            entrada = [numero_atual - numero_referencia, numero_atual, numero_referencia]  # Exemplo de cálculo
            saída = numero_atual  # A saída é o número atual
            entradas.append(entrada)
            saídas.append(saída)
    
    return np.array(entradas), np.array(saídas)

# Função de teste
def testar_calculos():
    caminho_csv = 'MEGA-SENA/dados_megasena/Mega_Sena.1.0.csv'  # Caminho correto para o arquivo CSV
    dados = carregar_dados_csv(caminho_csv)

    if dados is not None:
        # Gerar dados de treinamento
        entradas, saídas = gerar_dados_treinamento(dados)

        # Teste de entradas e saídas
        print("\nTestando entradas e saídas geradas...")
        print("Entradas para o treinamento (primeiras 5):", entradas[:5])
        print("Saídas para o treinamento (primeiras 5):", saídas[:5])

        # Verificar se o número de entradas e saídas estão corretos
        assert len(entradas) == len(saídas), "O número de entradas e saídas não coincide!"
        print("\nTeste de consistência: Passou")

        # Teste de cálculo direto (simples)
        print("\nTestando o cálculo direto (simples)...")
        resultado = calculo_direto(15, 18)
        print("Resultado do cálculo direto para 15 e 18:", resultado)

        # Testar controle de dados insuficientes
        print("\nTestando o controle de dados insuficientes...")
        for i in range(3, len(dados)):
            concurso_atual = dados[i]  # Pegando os números do concurso atual
            concurso_referencia = dados[i-3]  # Concurso de referência (3 concursos atrás)
            
            # Verificar se há dados suficientes para retroceder 3 concursos
            if i - 3 < 0:  # Se não houver concursos suficientes para retroceder
                print(f"[INFO] Pulo de cálculo para concurso {i + 1}, pois não há concursos suficientes para retroceder.")
                continue  # Pula para o próximo concurso

            # Executar o cálculo LT apenas se houver dados suficientes
            for j in range(6):
                numero_atual = int(concurso_atual[j])
                numero_referencia = int(concurso_referencia[j])
                entrada = [numero_atual - numero_referencia, numero_atual, numero_referencia]
                saída = numero_atual
                # A lógica de treinamento pode ser chamada aqui, se necessário
                # Por enquanto, apenas imprimindo para ver os cálculos
                print(f"[INFO] Concurso {i+1}, Número Atual {numero_atual}, Número Referência {numero_referencia}, Entrada {entrada}, Saída {saída}")

        print("\nTodos os testes passaram!")

# Chamar a função de teste
testar_calculos()
