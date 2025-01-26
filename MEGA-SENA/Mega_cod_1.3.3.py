import pandas as pd

# Função para carregar o CSV
def carregar_dados_csv():
    # Carregar o arquivo CSV
    dados = pd.read_csv("MEGA-SENA/dados_megasena/Mega_Sena.1.0.csv", delimiter=';')
    
    # Verificar e corrigir o formato das colunas
    dados.columns = dados.columns.str.strip()  # Remover espaços em branco
    return dados

# Função de cálculo LT
def calcular_lt_ajustado(concurso_atual, concurso_referencia):
    previsao_lt = []
    for i in range(6):
        # Calcula a previsão para cada número na sequência usando LT
        previsao = concurso_atual[i] - concurso_referencia[i]
        if previsao < 0:  # Se o valor for negativo, transforma em positivo
            previsao = abs(previsao)
        previsao_lt.append(previsao)
    return previsao_lt

# Função de cálculo reverso
def calcular_ajuste(concurso_atual, previsao_lt):
    ajustes = []
    for i in range(6):
        ajuste = concurso_atual[i] - previsao_lt[i]
        if ajuste > 60:  # Ajuste se exceder o limite de 60
            ajuste -= 6
        if ajuste < 1:  # Ajuste se for menor que 1
            ajuste += 6
        ajustes.append(ajuste)
    return ajustes

# Função principal para cálculo de previsões
# Função principal para cálculo de previsões ajustadas
def calcular_previsao_concursos_ajustado(dados):
    # Loop por todos os concursos, a partir do segundo (pois precisa de um concurso anterior)
    for i in range(6, len(dados)):
        # Concurso Atual (CA)
        concurso_atual = dados.iloc[i][['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values
        concurso_atual = [int(x) for x in concurso_atual]  # Converte para int simples

        # O número de concursos a voltar depende do primeiro número do concurso atual
        num_concursos_referencia = concurso_atual[0]
        linha_referencia = i - num_concursos_referencia
        
        if linha_referencia < 0:  # Evitar erros caso a linha de referência não exista
            print(f"[INFO] Não é possível calcular o concurso {i} (linha de referência inválida)")
            continue

        # Concurso Referência (CR) - A linha é calculada
        concurso_referencia = dados.iloc[linha_referencia][['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values
        concurso_referencia = [int(x) for x in concurso_referencia]  # Converte para int simples

        # Exibir os dados de CA e CR
        print(f"[INFO] Concurso Atual (Linha {i+1}): {concurso_atual}")
        print(f"[INFO] Concurso Referência (Linha {linha_referencia+1}): {concurso_referencia}")

        # Previsão LT para o concurso atual, baseado no concurso de referência
        previsao_lt = calcular_lt_ajustado(concurso_atual, concurso_referencia)
        print(f"[INFO] Previsão Completa para o Concurso {i+1}: {previsao_lt}")

        # Ajuste pelo cálculo reverso
        ajustes = calcular_ajuste(concurso_atual, previsao_lt)
        print(f"[INFO] Ajustes Realizados: {ajustes}")

        # Resultado esperado (linha seguinte ao concurso atual)
        concurso_esperado = dados.iloc[i + 1][['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values
        concurso_esperado = [int(x) for x in concurso_esperado]  # Converte para int simples
        print(f"[INFO] Resultado Esperado: {concurso_esperado}")

        # Para cada número, mostra a previsão, o esperado e o ajuste
        for j in range(6):
            print(f"[INFO] Previsão Calculada (LT) para o número {j + 1}: {previsao_lt[j]}")
            print(f"[INFO] Resultado Esperado: {concurso_esperado[j]}")
            print(f"[INFO] Ajuste Calculado pelo Cálculo Reverso: {ajustes[j]}")
            print("")

# Carregar os dados
dados = carregar_dados_csv()

# Calcular as previsões
calcular_previsao_concursos_ajustado(dados)
