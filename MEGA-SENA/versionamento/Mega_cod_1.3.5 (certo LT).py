import pandas as pd

def carregar_dados_csv():
    # Carregar o arquivo CSV
    dados = pd.read_csv("MEGA-SENA/dados_megasena/Mega_Sena.1.0.csv", delimiter=';')

    # Verificar e corrigir o formato das colunas
    dados.columns = dados.columns.str.strip()  # Remover espaços em branco
    return dados

def somar_digitos(numero):
    return sum(int(digito) for digito in str(numero))

def calcular_lt_ajustado(concurso_atual, concurso_referencia):
    previsao_lt = []
    for i in range(6):
        # Subtração direta entre números das sequências
        subtracao = concurso_atual[i] - concurso_referencia[i]
        
        # Soma dos dígitos dos números da sequência atual e de referência
        soma_digitos_atual = somar_digitos(concurso_atual[i])
        soma_digitos_referencia = somar_digitos(concurso_referencia[i])
        
        # Soma intermediária
        soma_total = soma_digitos_atual + soma_digitos_referencia

        # Soma final com a subtração
        previsao = soma_total + subtracao

        # Garantir que o valor esteja no intervalo de 1 a 60
        while previsao > 60:
            previsao -= 60
        while previsao < 1:
            previsao += 60

        previsao_lt.append(previsao)

    return previsao_lt

def calcular_previsao_concursos_ajustado(dados):
    for i in range(6, len(dados) - 1):
        # Concurso Atual (CA)
        concurso_atual = dados.iloc[i][['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values
        concurso_atual = [int(x) for x in concurso_atual]  # Converte para int simples

        # O número de concursos a voltar depende do primeiro número do concurso atual
        num_concursos_referencia = concurso_atual[0]
        linha_referencia = i - num_concursos_referencia

        if linha_referencia < 0:  # Evitar erros caso a linha de referência não exista
            print(f"[INFO] Não é possível calcular o concurso {i + 1} (linha de referência inválida)")
            continue

        # Concurso Referência (CR) - A linha é calculada
        concurso_referencia = dados.iloc[linha_referencia][['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values
        concurso_referencia = [int(x) for x in concurso_referencia]  # Converte para int simples

        # Exibir os dados de CA e CR
        print(f"[INFO] Concurso Atual (Linha {i + 1}): {concurso_atual}")
        print(f"[INFO] Concurso Referência (Linha {linha_referencia + 1}): {concurso_referencia}")

        # Previsão LT para o concurso atual, baseado no concurso de referência
        previsao_lt = calcular_lt_ajustado(concurso_atual, concurso_referencia)
        print(f"[INFO] Previsão Completa para o Concurso {i + 1}: {previsao_lt}")

        # Resultado esperado (linha seguinte ao concurso atual)
        concurso_esperado = dados.iloc[i + 1][['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values
        concurso_esperado = [int(x) for x in concurso_esperado]  # Converte para int simples
        print(f"[INFO] Resultado Esperado: {concurso_esperado}")

        # Para cada número, exibe a previsão e o resultado esperado
        for j in range(6):
            print(f"[INFO] Previsão Calculada (LT) para o número {j + 1}: {previsao_lt[j]}")
            print(f"[INFO] Resultado Esperado: {concurso_esperado[j]}")
            print("")

# Carregar os dados
dados = carregar_dados_csv()

# Calcular as previsões ajustadas
calcular_previsao_concursos_ajustado(dados)
