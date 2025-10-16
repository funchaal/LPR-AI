import cv2
import numpy as np

from modules.levenshtein import levenshtein
import logging

# Constantes para o ajuste de contraste e brilho da imagem da placa.
# ALPHA é o contraste (1 = sem alteração)
# BETA é o brilho (0 = sem alteração)
ALPHA = 1
BETA = 20


def post_process_plate(plate_frame):
    """
    Aplica um pós-processamento simples na imagem da placa para melhorar a visibilidade.
    Ajusta o contraste e o brilho da imagem.

    Args:
        plate_frame (numpy.ndarray): A imagem da placa cortada.

    Returns:
        numpy.ndarray: A imagem da placa com contraste e brilho ajustados.
    """
    # A função convertScaleAbs aplica a transformação: output = |alpha * input + beta|
    adjusted = cv2.convertScaleAbs(plate_frame, alpha=ALPHA, beta=BETA)

    return adjusted


def is_detection_stationary(tracker_state: dict, new_bbox: tuple, max_diff: int, stationary_frame_threshold: int) -> bool:
    """
    Verifica se uma detecção está estacionária usando um dicionário de estado independente.

    Args:
        tracker_state (dict): Dicionário para armazenar o estado.
                              Deve conter 'last_bbox' e 'stability_count'.
        new_bbox (tuple): A nova bounding box no formato (x1, y1, x2, y2).
        max_diff (int): A máxima variação em pixels para uma coordenada ser considerada estável.
        stationary_frame_threshold (int): Limiar de frames para considerar a placa estacionária.

    Returns:
        bool: True se a detecção for estacionária e deva ser ignorada.
              False se a detecção for nova ou tiver se movido.
    """
    last_bbox = tracker_state.get('last_bbox')

    # Se for a primeira detecção, inicializa o contador de estabilidade.
    if last_bbox is None:
        tracker_state['stability_count'] = 1
    else:
        # Compara a bounding box atual com a anterior.
        last_x1, last_y1, last_x2, last_y2 = last_bbox
        x1, y1, x2, y2 = new_bbox

        diff_x1 = abs(x1 - last_x1)
        diff_y1 = abs(y1 - last_y1)
        diff_x2 = abs(x2 - last_x2)
        diff_y2 = abs(y2 - last_y2)

        # Se a placa se moveu mais que o limite, reseta o contador.
        if not ((diff_x1 < max_diff and diff_y1 < max_diff) or
                (diff_x2 < max_diff and diff_y2 < max_diff)):
            tracker_state['stability_count'] = 1  # Resetou
        else:
            # Se continua estável, incrementa o contador.
            tracker_state['stability_count'] += 1

    # Atualiza o estado com a posição atual.
    tracker_state['last_bbox'] = new_bbox

    # Retorna True se a contagem ultrapassou o limiar, indicando que está estacionária.
    if tracker_state.get('stability_count', 0) > stationary_frame_threshold:
        return True

    return False


def choose_best_ocr_prediction(predictions):
    """
    Escolhe a melhor previsão de OCR de uma lista de detecções.
    O critério é a detecção com a maior área de bounding box.

    Args:
        predictions (list): Uma lista de previsões, onde cada item é uma tupla
                            contendo as coordenadas da caixa e o resultado do OCR (texto, confiança).

    Returns:
        tuple: O texto da placa e a pontuação de confiança da melhor previsão.
    """
    plate_text = ''
    score = 0
    max_area = 0

    # Itera sobre todas as previsões de OCR.
    for det in predictions:
        box_coords, (det_text, det_conf) = det
        # Calcula a área da bounding box da detecção.
        area = cv2.contourArea(np.array(box_coords, dtype=np.float32))

        # Se a área da detecção atual for maior que a maior área encontrada até agora,
        # atualiza a melhor previsão.
        if area > max_area:
            max_area = area
            plate_text = det_text
            score = det_conf

    return plate_text, score


import cv2


def crop_margin(frame, margin_percent):
    """
    Corta uma margem percentual da imagem para dentro.

    Args:
        frame (numpy.ndarray): Imagem OpenCV (BGR).
        margin_percent (int): Percentual da margem a cortar (0-100).

    Returns:
        numpy.ndarray: Imagem cropped.
    """
    if not 0 <= margin_percent <= 100:
        raise ValueError("margin_percent deve estar entre 0 e 100")

    h, w = frame.shape[:2]

    # Calcula pixels a cortar com base na porcentagem.
    margin_h = int(h * margin_percent / 100)
    margin_w = int(w * margin_percent / 100)

    # Realiza o corte na imagem.
    cropped_frame = frame[margin_h:h - margin_h, margin_w:w - margin_w]

    return cropped_frame


import re
from typing import List


class RegexValidator:
    """
    Classe para validar strings com base em uma expressão regular (regex).
    """

    def __init__(self, regex_pattern: str | None):
        """
        Inicializa o validador com um padrão de regex.

        Args:
            regex_pattern (str | None): O padrão de regex a ser usado para validação.
                                        Se for None, a validação sempre passará.
        """
        self.pattern_str = regex_pattern
        self.pattern = None

        if regex_pattern:
            try:
                # Compila o regex para um desempenho mais rápido.
                self.pattern = re.compile(regex_pattern)
                logging.info(f"Regex válido: {regex_pattern}")
            except re.error:
                logging.warning(f"Regex inválido fornecido: {regex_pattern}")
                # Se o regex for inválido, mantém o padrão como None e encerra.
                self.pattern = None
                exit(1)

    def validate(self, text: str) -> bool:
        """
        Valida a string com base no regex.
        Caso não haja regex válido, retorna sempre True.
        """
        if not self.pattern:
            return True
        return bool(self.pattern.match(text))

    def get_pattern(self) -> str | None:
        """Retorna o padrão de regex original como string."""
        return self.pattern_str

    def filter_list(self, items: List[str]) -> List[str]:
        """
        Retorna apenas os elementos válidos de uma lista.
        Se não houver regex válido, retorna a lista inteira.
        """
        if not self.pattern:
            return items
        return [item for item in items if self.validate(item)]


from collections import defaultdict


def definePossibleReadings(plates: dict) -> list:
    """
    Algoritmo para pontuar e classificar as leituras de placas.
    A pontuação é baseada na frequência de substrings em comum entre as leituras.
    """
    if not plates:
        return []

    plate_pontuation = defaultdict(int)

    # Itera sobre cada placa lida.
    for plate in plates.keys():
        # Gera todas as substrings possíveis da placa com tamanho de 2 ou mais.
        substrings = {
            plate[j:j + i]
            for i in range(2, len(plate) + 1)
            for j in range(len(plate) - i + 1)
        }
        # Para cada substring, verifica se ela aparece em outras leituras.
        for substring in substrings:
            for reading, count in plates.items():
                if substring in reading:
                    # Aumenta a pontuação da placa original com base na contagem da leitura correspondente.
                    plate_pontuation[plate] += plates[plate]

    # Retorna as 2 placas com a maior pontuação.
    top_plates = sorted(
        plate_pontuation, key=plate_pontuation.get, reverse=True)[:2]
    return top_plates


from itertools import product
from typing import List, Dict


class FormatConverter:
    """
    Converte leituras de placas para formatos especificados, aplicando correções de caracteres.
    """

    def __init__(self, formats: List[str], char_corrections: Dict[str, List[str]]):
        """
        :param formats: lista de formatos (ex: ["LLLNNNN"])
        :param char_corrections: dicionário de substituições (ex: {"0": ["O"], "O": ["0"]})
        """
        self.formats = formats
        self.char_corrections = char_corrections
        self.formats_by_len = self._group_formats_by_length()

    @staticmethod
    def _is_letter(c: str) -> bool:
        """Verifica se um caractere é uma letra."""
        return c.isalpha()

    @staticmethod
    def _is_digit(c: str) -> bool:
        """Verifica se um caractere é um dígito."""
        return c.isdigit()

    @classmethod
    def _fits_token(cls, ch: str, token: str) -> bool:
        """Verifica se um caractere corresponde a um token de formato ('L' para letra, 'N' para número)."""
        if token == "L":
            return cls._is_letter(ch)
        if token == "N":
            return cls._is_digit(ch)
        raise ValueError(f"Token inválido: {token}")

    def _group_formats_by_length(self) -> Dict[int, List[str]]:
        """Organiza os formatos por tamanho para uma busca rápida."""
        grouped: Dict[int, List[str]] = {}
        for f in self.formats:
            grouped.setdefault(len(f), []).append(f)
        return grouped

    def convert(self, readings: List[str]) -> Dict[str, List[str]]:
        """
        Converte leituras de placas para os formatos especificados,
        aplicando correções de caracteres.
        """
        resultados: Dict[str, List[str]] = {}

        for reading in readings:
            # Obtém os formatos correspondentes ao comprimento da leitura.
            fmt_list = self.formats_by_len.get(len(reading), [])
            if not fmt_list:
                continue

            variantes_set = set()

            for fmt in fmt_list:
                mismatch_positions, candidate_lists = [], []
                ok_entire = True

                # Verifica cada caractere da leitura em relação ao formato.
                for i, tok in enumerate(fmt):
                    ch = reading[i]
                    if self._fits_token(ch, tok):
                        continue

                    ok_entire = False
                    # Se não corresponder, busca por correções possíveis.
                    cand = self.char_corrections.get(ch)
                    if not cand:
                        candidate_lists = []
                        break

                    # Filtra as correções que se encaixam no token do formato.
                    cand_ok = [c for c in cand if self._fits_token(c, tok)]
                    if not cand_ok:
                        candidate_lists = []
                        break

                    mismatch_positions.append(i)
                    candidate_lists.append(cand_ok)

                if ok_entire or not candidate_lists:
                    continue

                # Gera todas as combinações de variantes da placa com as correções.
                base = list(reading)
                for combo in product(*candidate_lists):
                    buff = base[:]
                    for pos, val in zip(mismatch_positions, combo):
                        buff[pos] = val
                    variantes_set.add("".join(buff))

            if variantes_set:
                resultados[reading] = list(variantes_set)

        return resultados


def chooseBestFrame(frames_data: list, comparison_text: str = None) -> dict | None:
    """
    Seleciona o melhor frame com uma lógica de fallback progressiva.
    """
    if not frames_data:
        return None

    # Se não há uma leitura final para comparação, usa o critério de centralização em todos os frames.
    if not comparison_text:
        logging.debug(f"Sem leitura final, usando critério de centralização")
        candidate_frames = frames_data
    else:
        candidate_frames = []
        # Tenta encontrar frames com leituras próximas à leitura final,
        # aumentando a tolerância da distância de Levenshtein.
        for max_distance in [2, 3, 4]:
            for frame in frames_data:
                try:
                    reading_text = frame.get('reading', '')
                    if levenshtein(comparison_text, reading_text) <= max_distance:
                        candidate_frames.append(frame)
                except Exception as e:
                    logging.warning(f"Erro ao processar frame: {e}")
                    continue

            if candidate_frames:
                break

        # Se nenhum frame for encontrado com a distância de Levenshtein,
        # usa todos os frames como candidatos.
        if not candidate_frames:
            logging.debug(
                f"Nenhum frame com Levenshtein <= 4. Usando todos.")
            candidate_frames = frames_data

    # Dentre os frames candidatos, encontra o mais centralizado na imagem.
    best_frame = None
    min_distance = float('inf')

    for frame in candidate_frames:
        try:
            height, width = frame['input_frame'].shape[:2]
            image_center_x, image_center_y = width / 2, height / 2

            x1, y1, x2, y2 = frame['bounding_box']
            bbox_center_x, bbox_center_y = (x1 + x2) / 2, (y1 + y2) / 2

            # Calcula a distância euclidiana do centro da bounding box ao centro da imagem.
            distance = ((bbox_center_x - image_center_x) ** 2 +
                        (bbox_center_y - image_center_y) ** 2) ** 0.5

            # O frame com a menor distância é considerado o melhor.
            if distance < min_distance:
                min_distance = distance
                best_frame = frame
        except (KeyError, TypeError) as e:
            logging.warning(f"Frame malformado: {e}")
            continue

    return best_frame
