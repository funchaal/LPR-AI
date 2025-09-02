def levenshtein(s1: str, s2: str) -> int:
    """
    Calcula a distância de Levenshtein entre duas strings (s1 e s2).

    A distância de Levenshtein é o número mínimo de edições de um único caractere
    (inserções, exclusões ou substituições) necessárias para transformar uma
    string na outra.

    Args:
        s1 (str): A primeira string.
        s2 (str): A segunda string.

    Returns:
        int: A distância de Levenshtein entre s1 e s2.
    """
    # Garante que s1 seja a string menor para otimizar o espaço da matriz
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # Se s2 for uma string vazia, a distância é o tamanho de s1 (só inserções)
    if not s2:
        return len(s1)

    # Inicializa a linha anterior da matriz de distâncias
    # O +1 é para a string vazia no início
    previous_row = list(range(len(s2) + 1))

    for i, c1 in enumerate(s1):
        # A linha atual começa com a distância de deleção (i + 1)
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Calcula o custo de cada operação
            insercoes = previous_row[j + 1] + 1
            delecoes = current_row[j] + 1
            # O custo de substituição é 0 se os caracteres forem iguais, 1 caso contrário
            substituicoes = previous_row[j] + (c1 != c2)
            
            # A célula atual recebe o menor custo entre as três operações
            current_row.append(min(insercoes, delecoes, substituicoes))
        
        # A linha atual se torna a linha anterior para a próxima iteração
        previous_row = current_row

    # O resultado final está na última célula da última linha
    return previous_row[-1]