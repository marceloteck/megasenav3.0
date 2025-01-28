# Código redefinido devido à reinicialização. Vou reimplementar e calcular novamente.

import numpy as np

# Funções
def ajustar_repeticoes(resultado_ajustado):
    """Ajusta números repetidos na sequência."""
    usados = set()
    ajustado_final = []
    for numero in resultado_ajustado:
        if numero not in usados:
            ajustado_final.append(numero)
            usados.add(numero)
        else:
            # Substitui o número repetido por um não utilizado (1 a 60)
            for i in range(1, 61):
                if i not in usados:
                    ajustado_final.append(i)
                    usados.add(i)
                    break
    return np.array(ajustado_final)


def calcular_com_raiz_pi(sequencia_1, sequencia_2):
    """Calcula novos números com base em raiz quadrada e pi."""
    # Combina as duas sequências em uma única referência
    sequencia_referencia = (sequencia_1 + sequencia_2) / 2  # Média simples das duas sequências
    
    # Aplica a fórmula: sqrt(n) * pi e ajusta para o intervalo de 1 a 60
    novos_numeros = np.sqrt(sequencia_referencia) * np.pi
    novos_numeros = np.round(novos_numeros % 60).astype(int)  # Ajuste para o intervalo de 1 a 60
    
    # Ajustar números repetidos
    return ajustar_repeticoes(novos_numeros)


# Sequências fornecidas
sequencia_1 = np.array([13, 14, 31, 33, 35, 43])
sequencia_2 = np.array([5, 11, 14, 35, 53, 56])

# Calculando a nova sequência
nova_sequencia_com_raiz_pi = calcular_com_raiz_pi(sequencia_1, sequencia_2)

nova_sequencia_com_raiz_pi
