import cv2
import numpy as np

from modules.levenshtein import levenshtein
import logging

ALPHA = 1
BETA = 20

def post_process_plate(plate_frame):
    adjusted = cv2.convertScaleAbs(plate_frame, alpha=ALPHA, beta=BETA)

    return adjusted

def choose_best_ocr_prediction(predictions):
    plate_text = ''
    score = 0
    max_area = 0

    for det in predictions:
        box_coords, (det_text, det_conf) = det
        area = cv2.contourArea(np.array(box_coords, dtype=np.float32))

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

    # Calcula pixels a cortar
    margin_h = int(h * margin_percent / 100)
    margin_w = int(w * margin_percent / 100)

    # Crop
    cropped_frame = frame[margin_h:h - margin_h, margin_w:w - margin_w]

    return cropped_frame

import re
from typing import List

class RegexValidator:
    def __init__(self, regex_pattern: str | None):
        self.pattern_str = regex_pattern
        self.pattern = None

        if regex_pattern:
            try:
                self.pattern = re.compile(regex_pattern)
                logging.info(f"Regex válido: {regex_pattern}")
            except re.error:
                logging.warning(f"Regex inválido fornecido: {regex_pattern}")
                # Se o regex for inválido, ignora e mantém None
                self.pattern = None

    def validate(self, text: str) -> bool:
        """
        Valida a string com base no regex.
        Caso não haja regex válido, retorna sempre True.
        """
        if not self.pattern:
            return True
        return bool(self.pattern.match(text))

    def get_pattern(self) -> str | None:
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
    """
    if not plates:
        return []
        
    plate_pontuation = defaultdict(int)

    for plate in plates.keys():
        substrings = {
            plate[j:j + i]
            for i in range(2, len(plate) + 1)
            for j in range(len(plate) - i + 1)
        }
        for substring in substrings:
            for reading, count in plates.items():
                if substring in reading:
                    plate_pontuation[plate] += plates[plate]

    top_plates = sorted(plate_pontuation, key=plate_pontuation.get, reverse=True)[:2]
    return top_plates

from itertools import product
from typing import List, Dict

class FormatConverter:
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
        return c.isalpha()

    @staticmethod
    def _is_digit(c: str) -> bool:
        return c.isdigit()

    @classmethod
    def _fits_token(cls, ch: str, token: str) -> bool:
        if token == "L":
            return cls._is_letter(ch)
        if token == "N":
            return cls._is_digit(ch)
        raise ValueError(f"Token inválido: {token}")

    def _group_formats_by_length(self) -> Dict[int, List[str]]:
        """Organiza formatos por tamanho, para lookup rápido."""
        grouped: Dict[int, List[str]] = {}
        for f in self.formats:
            grouped.setdefault(len(f), []).append(f)
        return grouped

    def convert(self, readings: List[str]) -> Dict[str, List[str]]:
        """
        Converte leituras de placas para formatos especificados,
        aplicando correções de caracteres.
        """
        resultados: Dict[str, List[str]] = {}

        for reading in readings:
            fmt_list = self.formats_by_len.get(len(reading), [])
            if not fmt_list:
                continue

            variantes_set = set()

            for fmt in fmt_list:
                mismatch_positions, candidate_lists = [], []
                ok_entire = True

                for i, tok in enumerate(fmt):
                    ch = reading[i]
                    if self._fits_token(ch, tok):
                        continue

                    ok_entire = False
                    cand = self.char_corrections.get(ch)
                    if not cand:
                        candidate_lists = []
                        break

                    cand_ok = [c for c in cand if self._fits_token(c, tok)]
                    if not cand_ok:
                        candidate_lists = []
                        break

                    mismatch_positions.append(i)
                    candidate_lists.append(cand_ok)

                if ok_entire or not candidate_lists:
                    continue

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

        # Se não há leitura final, usa critério de centralização
        if not comparison_text:
            logging.debug(f"Sem leitura final, usando critério de centralização")
            candidate_frames = frames_data
        else:
            candidate_frames = []
            # Tenta diferentes distâncias de Levenshtein
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

            # Fallback: usa todos os frames se não encontrou nenhum
            if not candidate_frames:
                logging.debug(f"Nenhum frame com Levenshtein <= 4. Usando todos.")
                candidate_frames = frames_data

        # Encontra o frame mais centralizado
        best_frame = None
        min_distance = float('inf')

        for frame in candidate_frames:
            try:
                height, width = frame['input_frame'].shape[:2]
                image_center_x, image_center_y = width / 2, height / 2

                x1, y1, x2, y2 = frame['plate_bounding_box']
                bbox_center_x, bbox_center_y = (x1 + x2) / 2, (y1 + y2) / 2
                
                distance = ((bbox_center_x - image_center_x) ** 2 + (bbox_center_y - image_center_y) ** 2) ** 0.5

                if distance < min_distance:
                    min_distance = distance
                    best_frame = frame
            except (KeyError, TypeError) as e:
                logging.warning(f"Frame malformado: {e}")
                continue

        return best_frame