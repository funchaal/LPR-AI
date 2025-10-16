def validate_bounding_box(x1, y1, x2, y2):
    """
    Valida as dimensões de uma bounding box.

    Args:
        x1 (int): Coordenada x do canto superior esquerdo.
        y1 (int): Coordenada y do canto superior esquerdo.
        x2 (int): Coordenada x do canto inferior direito.
        y2 (int): Coordenada y do canto inferior direito.

    Returns:
        bool: True se a proporção da bounding box for válida, False caso contrário.
    """
    # Calcula a largura e a altura da bounding box
    width = x2 - x1
    height = y2 - y1

    # Verifica se a proporção da largura pela altura é menor que 16/9
    # Esta é uma verificação para garantir que a bounding box se assemelha a uma placa de carro
    if width / height < 16/9:
        return False
    else:
        return True

def validate_text(text, min_length=3):
    """
    Valida um texto, verificando se não é composto apenas por dígitos, apenas por letras ou se é muito curto.

    Args:
        text (str): O texto a ser validado.
        min_length (int, optional): O comprimento mínimo que o texto deve ter. Defaults to 3.

    Returns:
        bool: True se o texto for válido, False caso contrário.
    """
    def is_all_digits(s):
        """Verifica se a string contém apenas dígitos."""
        return s.isdigit()

    def is_all_letters(s):
        """Verifica se a string contém apenas letras."""
        return s.isalpha()

    def is_too_short(s, min_len):
        """Verifica se a string é mais curta que o comprimento mínimo."""
        return len(s) <= min_len

    # Se o texto for composto apenas por dígitos, ou apenas por letras, ou for muito curto, a validação falha
    if is_all_digits(text) or is_all_letters(text) or is_too_short(text, min_length):
        return False
    # Caso contrário, o texto é considerado válido
    return True