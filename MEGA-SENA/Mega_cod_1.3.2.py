import pandas as pd

# Função para carregar os dados do arquivo CSV com delimitador ';'
def carregar_dados_csv(caminho_csv):
    try:
        dados = pd.read_csv(caminho_csv, delimiter=';')  # Especificando o delimitador
        print(f"[INFO] Colunas disponíveis: {dados.columns}")  # Exibindo as colunas para diagnóstico
        return dados
    except Exception as e:
        print(f"[ERROR] Não foi possível carregar os dados: {e}")
        return None

# Função para realizar o cálculo direto
def calculo_direto(numero_atual, numero_referencia):
    resultado = numero_atual - numero_referencia
    resultado_soma = sum(map(int, str(abs(numero_atual)))) + sum(map(int, str(abs(numero_referencia))))
    return resultado_soma + resultado

# Função para realizar o cálculo reverso
def calculo_reverso(numero_atual, numero_referencia):
    resultado = calculo_direto(numero_referencia, numero_atual)
    return resultado

# Função para calcular a previsão de todos os concursos
def calcular_previsao_concursos(dados):
    for i in range(6, len(dados)):  # Começa do 6º concurso (pois precisa de 5 anteriores para calcular)
        # Ajuste para garantir que os nomes das colunas estão corretos
        try:
            # Ajustando a exibição dos números
            concurso_atual = dados.iloc[i][['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values
            concurso_atual = [int(x) for x in concurso_atual]  # Converte np.int64 para int simples

            concurso_referencia = dados.iloc[i - 1][['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values
            concurso_referencia = [int(x) for x in concurso_referencia]  # Converte np.int64 para int simples

        except KeyError as e:
            print(f"[ERROR] Colunas não encontradas: {e}. Verifique o nome das colunas no CSV.")
            continue
        
        # Verifica se há dados suficientes para o cálculo (pelo menos 5 concursos anteriores)
        if i < 6:
            print(f"[INFO] Pulo de cálculo para concurso {i+1}, pois não há concursos suficientes para retroceder.")
            continue
        
        # Aplica o cálculo LT para todos os números da sequência
        sequencia_resultado = []
        sequencia_esperada = []
        ajustes = []
        
        for j in range(6):
            numero_atual = int(concurso_atual[j])
            numero_referencia = int(concurso_referencia[j])
            
            resultado_previsao = calculo_direto(numero_atual, numero_referencia)
            sequencia_resultado.append(resultado_previsao)
            
            # Resultado esperado e cálculo reverso
            resultado_esperado = concurso_atual[j]
            resultado_ajuste = calculo_reverso(numero_atual, numero_referencia)
            ajustes.append(resultado_ajuste)
            
            # Exibindo o que foi calculado
            print(f"\n[INFO] Concurso Atual (Linha {i+1}): {concurso_atual}")
            print(f"[INFO] Concurso Referência (Linha {i}): {concurso_referencia}")
            print(f"[INFO] Previsão Calculada (LT) para o número {j+1}: {resultado_previsao}")
            print(f"[INFO] Resultado Esperado: {resultado_esperado}")
            print(f"[INFO] Ajuste Calculado pelo Cálculo Reverso: {resultado_ajuste}")
        
        print(f"\n[INFO] Previsão Completa para o Concurso {i+1}: {sequencia_resultado}")
        print(f"[INFO] Ajustes Realizados: {ajustes}")
        print("="*50)

# Caminho do arquivo CSV
caminho_csv = 'MEGA-SENA/dados_megasena/Mega_Sena.1.0.csv'

# Carregar dados
dados = carregar_dados_csv(caminho_csv)

if dados is not None:
    calcular_previsao_concursos(dados)
