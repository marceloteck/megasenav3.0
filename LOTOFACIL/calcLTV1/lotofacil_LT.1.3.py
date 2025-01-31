import csv
import os
from sklearn.linear_model import LinearRegression
import numpy as np

# Função para carregar dados do CSV
def carregar_dados_csv(caminho_csv):
    dados = []
    with open(caminho_csv, newline='', encoding='utf-8') as arquivo_csv:
        leitor = csv.reader(arquivo_csv, delimiter=';')
        next(leitor, None)  # Ignora o cabeçalho, se existir

        for linha in leitor:
            try:
                dados.append([int(n) for n in linha[:15]])  # Considera os 15 primeiros números da Lotofácil
            except ValueError:
                # Ignorar linhas mal formatadas
                continue

    # Filtrar apenas sequências com 15 números
    dados = [linha for linha in dados if len(linha) == 15]
    return dados

# Função para calcular ajustes entre sequências
def calcular_ajustes_entre_sequencias(sequencia_atual, sequencia_referencia):
    return [sequencia_referencia[i] - sequencia_atual[i] for i in range(15)]

# Função para salvar os ajustes no arquivo TXT
def salvar_historico_ajustes(previsao_lt, ajustes):
    historico_path = "historico_ajustes_lotofacil.txt"
    with open(historico_path, "a") as arquivo:
        arquivo.write(f"Previsao: {previsao_lt}, Ajustes: {ajustes}\n")

# Função para treinar o modelo
def treinar_modelo(ajustes_hist):
    X = []
    y = []

    # Considerando ajustes anteriores como entrada para prever o próximo ajuste
    for i in range(len(ajustes_hist) - 1):
        X.append(ajustes_hist[i])  # Ajuste atual
        y.append(ajustes_hist[i + 1])  # Ajuste seguinte (a ser previsto)

    if len(X) == 0:
        print("Dados insuficientes para treinar o modelo.")
        return None

    # Convertendo para numpy arrays para o modelo
    X = np.array(X)
    y = np.array(y)

    modelo = LinearRegression()
    modelo.fit(X, y)

    return modelo

# Função para prever três opções de ajustes
def prever_tres_opcoes_ajustes(modelo, ultimo_ajuste):
    previsao_central = modelo.predict([ultimo_ajuste])[0].astype(int).tolist()

    # Gerar opções variando a previsão central
    margem_variacao = 1  # Pequena variação nos ajustes

    previsao_opcao1 = [(n + margem_variacao) % 25 for n in previsao_central]  # Ajuste positivo
    previsao_opcao2 = [(n - margem_variacao) % 25 for n in previsao_central]  # Ajuste negativo

    return [previsao_central, previsao_opcao1, previsao_opcao2]

# Função para calcular a sequência prevista com base nos ajustes
def calcular_sequencia_prevista(ultimo_sorteio, ajustes_previstos):
    return [(ultimo_sorteio[i] + ajustes_previstos[i] - 1) % 25 + 1 for i in range(15)]

# Nova função para formar uma sequência com base nas três opções mantendo aprendizado do modelo
def calcular_nova_sequencia(opcoes_ajustes, modelo):
    # Combina todas as opções de ajustes e aplica aprendizado do modelo para priorizar padrões fortes
    todas_previsoes = np.array(opcoes_ajustes).flatten()

    # Frequência de cada número válido (1 a 25)
    numeros_unicos, frequencias = np.unique(todas_previsoes, return_counts=True)

    # Filtrar para manter números dentro da faixa válida (1 a 25)
    numeros_validos = [(num, freq) for num, freq in zip(numeros_unicos, frequencias) if 1 <= num <= 25]

    # Ordenar por frequência decrescente, mas ponderando pelo modelo
    numeros_validos.sort(key=lambda x: -x[1])

    # Aplicar peso para números com base na previsão central
    previsao_base = np.array(opcoes_ajustes[0])  # Base na primeira opção
    pesos = modelo.predict([previsao_base])[0].tolist()
    pesos = [max(1, int(peso)) for peso in pesos]

    # Associar pesos aos números válidos
    numeros_com_pesos = [(num, freq * pesos[idx % len(pesos)]) for idx, (num, freq) in enumerate(numeros_validos)]

    # Ordenar por relevância ponderada
    numeros_com_pesos.sort(key=lambda x: -x[1])

    # Pegar os 15 melhores números
    nova_sequencia = [num for num, _ in numeros_com_pesos[:15]]
    return sorted(nova_sequencia)

# Função principal para processar dados e armazenar histórico
def processar_dados_e_armazenar_ajustes(caminho_csv):
    if not os.path.exists(caminho_csv):
        print("Arquivo CSV não encontrado!")
        return

    dados_csv = carregar_dados_csv(caminho_csv)

    if len(dados_csv) < 2:
        print("Poucos dados para calcular ajustes.")
        return

    ajustes_hist = []
    for i in range(len(dados_csv) - 1):
        previsao_lt = dados_csv[i]
        sequencia_real = dados_csv[i + 1]
        ajustes = calcular_ajustes_entre_sequencias(previsao_lt, sequencia_real)
        salvar_historico_ajustes(previsao_lt, ajustes)
        ajustes_hist.append(ajustes)

    print("Histórico de ajustes salvo com sucesso.")

    modelo = treinar_modelo(ajustes_hist)

    if modelo is None:
        return

    ultimo_ajuste = ajustes_hist[-1]
    previsoes_ajustes = prever_tres_opcoes_ajustes(modelo, ultimo_ajuste)

    print(f"\nTrês opções de previsões de ajustes para o próximo sorteio: {previsoes_ajustes}")

    ultimo_sorteio = dados_csv[-1]
    print("\nNúmeros previstos com as três opções de ajustes:")

    opcoes_sequencias = []
    for i, ajustes_previstos in enumerate(previsoes_ajustes, 1):
        sequencia_prevista = calcular_sequencia_prevista(ultimo_sorteio, ajustes_previstos)
        print(f"Opção {i}: {list(map(int, sorted(sequencia_prevista)))}")
        opcoes_sequencias.append(sequencia_prevista)

    # Gerar a nova sequência baseada nas três opções com aprendizado do modelo
    nova_sequencia = calcular_nova_sequencia(opcoes_sequencias, modelo)
    print(f"\nNova sequência com maior probabilidade de acerto: {list(map(int, nova_sequencia))}")


# Caminho do arquivo CSV
caminho_do_csv = "lotofacil.csv"

processar_dados_e_armazenar_ajustes(caminho_do_csv)
