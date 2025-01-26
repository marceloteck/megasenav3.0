# Importando biblioteca para configuração do ambiente
import os

# Configuração do ambiente TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Imports essenciais
import numpy as np  # Para manipulação de arrays e cálculos numéricos
import pandas as pd  # Para manipulação de dados (caso seja necessário mais tarde)

# Para rede neural (usaremos mais adiante)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Para testes e validação
from sklearn.model_selection import train_test_split

# Função de cálculo direto
def calculo_direto(numero1, numero2):
    # 1. Subtração entre os números
    subtracao = numero1 - numero2
    
    # 2. Soma dos dígitos de cada número
    soma_digitos_1 = sum([int(digit) for digit in str(abs(numero1))])
    soma_digitos_2 = sum([int(digit) for digit in str(abs(numero2))])
    
    # 3. Soma da subtração com a soma dos dígitos
    resultado_intermediario = soma_digitos_1 + soma_digitos_2 + subtracao
    
    return resultado_intermediario

# Função de cálculo reverso
def calculo_reverso(resultado_intermediario, numero_final):
    # 1. Identificar a diferença entre o resultado intermediário e o número final
    ajuste = numero_final - resultado_intermediario
    
    # 2. Ajustar o valor dentro do intervalo de 1 a 60, se necessário
    while ajuste > 60:
        ajuste -= 3  # Ajuste para ficar dentro do intervalo
    
    while ajuste < 1:
        ajuste += 3  # Ajuste para ficar dentro do intervalo
    
    return ajuste

# Teste de cálculo com números de exemplo
numero1 = 15  # Exemplo da sequência 2
numero2 = 4   # Exemplo da sequência 1

# Cálculo direto
resultado_intermediario = calculo_direto(numero1, numero2)
print("Resultado Intermediário do Cálculo Direto:", resultado_intermediario)

# Número final (resultado esperado)
numero_final = 24  # Resultado final do concurso

# Cálculo reverso
ajuste = calculo_reverso(resultado_intermediario, numero_final)
print("Ajuste calculado no Cálculo Reverso:", ajuste)
