def adaptar_sequencia(op1, op2, op3):
    nova_sequencia = []
    usados = set()  # Conjunto para rastrear valores usados e evitar repetições

    for i in range(len(op1)):
        # Condição 6: Se op1[i], op2[i] e op3[i] estão em sequência crescente (ex: 19, 20, 21)
        if i < len(op1) - 1 and op1[i] + 1 == op2[i] and op2[i] + 1 == op3[i]:
            nova_sequencia.append(op1[i])  # Mantém o valor de op1[i]
            usados.add(op1[i])
            # Adiciona o valor de op3[i + 1] ou o próximo valor válido
            if i + 1 < len(op1):
                proximo = op3[i] if i + 1 < len(op3) else op1[i + 1]
                if proximo not in usados:
                    nova_sequencia.append(proximo)
                    usados.add(proximo)
            continue

        # Condição: Se op1[i] == 16 e a sequência anterior for 15 nos dois outros arrays
        if i > 0 and op1[i] == 16 and op2[i - 1] == 15 and op3[i - 1] == 15 and op2[i] != op1[i]:
            # Remover o 16 e adicionar o próximo valor válido
            if i + 1 < len(op1):
                proximo = op1[i + 1]
                if proximo not in usados:
                    nova_sequencia.append(proximo)
                    usados.add(proximo)
            continue

        # Condição 5: Sequência crescente entre op1, op2 e op3
        if i < len(op1) - 2 and op1[i] + 1 == op2[i] and op2[i] + 1 == op3[i]:
            nova_sequencia.append(op1[i])
            usados.add(op1[i])
            # Move o valor de op3[i] para o próximo op1
            if i + 1 < len(op1):
                proximo = op3[i]
                if proximo not in usados:
                    nova_sequencia.append(proximo)
                    usados.add(proximo)
                # Remove o próximo op1 e adiciona ele na posição seguinte
                if i + 2 < len(op1):
                    outro_proximo = op1[i + 1]
                    if outro_proximo not in usados:
                        nova_sequencia.append(outro_proximo)
                        usados.add(outro_proximo)
            continue

        # Condição 4: Se op2[i - 1] e op3[i - 1] forem ambos 13
        if i > 0 and op2[i - 1] == 13 and op3[i - 1] == 13:
            if 13 not in usados:
                nova_sequencia.append(13)
                usados.add(13)
        # Condição 3: Se op2[i - 1] e op3[i - 1] forem ambos 10
        elif i > 0 and op2[i - 1] == 10 and op3[i - 1] == 10:
            if 10 not in usados:
                nova_sequencia.append(10)
                usados.add(10)
        # Condição 2: Se op1 e op3 forem iguais, e op2 for diferente
        elif op1[i] == op3[i] and op2[i] != op1[i]:
            if op1[i] not in usados:
                nova_sequencia.append(op1[i])
                usados.add(op1[i])
        # Condição 1: Se op2 e op3 forem iguais
        elif op2[i] == op3[i]:
            if i > 0 and op1[i-1] + 1 == op2[i-1] and op2[i-1] + 1 == op3[i-1]:
                usados.add(op3[i+1])  # Adiciona o próximo número de op3
            elif op1[i] not in usados:
                nova_sequencia.append(op1[i])
                usados.add(op1[i])

            if op1[i] == 24 and op3[i] == 25:
                usados.add(25)  # Adiciona 25 diretamente
        else:
            # Mantém op1 como escolha padrão
            if op1[i] not in usados:
                nova_sequencia.append(op1[i])
                usados.add(op1[i])

    # Garantir a sequência limpa (limitada ao tamanho original) e sem repetições
    while len(nova_sequencia) < len(op1):
        for num in range(1, 26):  # Supondo que os números variem de 1 a 25
            if num not in usados:
                nova_sequencia.append(num)
                usados.add(num)
                break

    return nova_sequencia[:len(op1)]

# Exemplo de opções
op1 = [1, 3, 5, 6, 8, 9, 11, 12, 14, 16, 17, 18, 19, 22, 24]
op2 = [2, 4, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 20, 23, 25]
op3 = [2, 4, 5, 7, 8, 10, 11, 13, 15, 16, 17, 18, 21, 23, 25]

# Adaptando a sequência
nova_sequencia = adaptar_sequencia(op1, op2, op3)
print("Nova sequência adaptada:", nova_sequencia)



# cd LOTOFACIL/calcLTV1
# python CALC_NV.py

#[1 3 5 6 8 9 10 12 13 17 18 19 21 24 25]