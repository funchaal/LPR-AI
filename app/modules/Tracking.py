# tracking.py

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import logging
import uuid
import cv2
import requests 
import threading

from modules.levenshtein import levenshtein

from itertools import product
from typing import Dict, List

class Tracking:
    """
    Rastreia a passagem de um objeto (ex: veículo) através dos frames, 
    coletando leituras (ex: placas) e determinando a melhor leitura final.
    """
    
    # --- Variáveis de Classe para Configuração e Estado Global ---
    trackings = {}
    suspect_detections = []
    reading_formats = []
    char_corrections: Dict[str, List[str]] = {}
    db_manager = None
    instance_id = None
    captures_save_path = None
    suspect_detections_save_path = None
    use_continuous_tries = False
    api_endpoint = None

    # --- Métodos de Instância (para cada passagem individual) ---

    def __init__(self):
        """Inicializa um novo objeto de rastreamento."""
        self.start_time = datetime.now()
        self.id = str(uuid.uuid4())
        self.readings = defaultdict(int)
        self.possibleReadings = []
        self.finalReading = ''
        self.noFrameCount = 0
        self.frames = []
        self.is_closing = False

    def __str__(self):
        """Representação em string do objeto de rastreamento."""
        return f'ID: {self.id}, Placas Possíveis: {self.possibleReadings}, Placa Final: {self.finalReading}'

    def addCapture(self, reading: str, frame_data: dict):
        """
        Adiciona uma nova leitura e os dados do frame a este rastreamento.
        """
        self.readings[reading] += 1
        self.frames.append(frame_data)
        self.noFrameCount = 0

        if self.__class__.use_continuous_tries:
            self._update_and_call_api()

    def _call_api_async(self, payload: dict):
        """
        Função alvo da thread para chamar a API de forma assíncrona.
        """
        try:
            logging.info(f"Enviando dados para API para captura {self.id}: {payload['readings']}")
            response = requests.post(self.__class__.api_endpoint, json=payload, timeout=10)

            track_instance = self.__class__.trackings.get(self.id)
            if not track_instance:
                logging.warning(f"API retornou para a captura {self.id}, mas ela já foi fechada.")
                return

            if response.status_code == 200:
                correct_reading = response.text
                logging.info(f"API retornou 200 para captura {self.id}. Leitura correta: {correct_reading}")
                track_instance.finalReading = correct_reading
                track_instance.close()
            
            elif response.status_code == 204:
                logging.info(f"API retornou 204 para captura {self.id}. Nenhuma leitura correspondeu. Continuando...")
            
            else:
                logging.error(f"Erro na API para captura {self.id}. Status: {response.status_code}, Resposta: {response.text}")

        except requests.exceptions.RequestException as e:
            logging.error(f"Falha ao chamar a API para captura {self.id}: {e}")

    def _update_and_call_api(self):
        """Verifica novas placas possíveis e chama a API externa, se configurado."""

        if self.__class__.reading_formats:
            original_readings = list(self.readings.keys())
            converted_readings_dict = self.convert_readings_to_formats(self.__class__.reading_formats, original_readings, self.__class__.char_corrections)
            
            logging.info(f"Placas convertidas: {converted_readings_dict}")
            
            updated_readings = self.readings.copy()
            
            for original_plate, converted_readings in converted_readings_dict.items():
                original_score = self.readings[original_plate]
                for converted_plate in converted_readings:
                    updated_readings[converted_plate] = updated_readings.get(converted_plate, 0) + original_score
            
            self.readings = updated_readings

        defined_possible_readings = self.definePossibleReadings(self.readings)
        
        # --- LÓGICA ORIGINAL RESTAURADA ---
        # Encontra novas leituras possíveis mantendo seu índice original para inserção
        new_possible_readings = [[idx, x] for idx, x in enumerate(defined_possible_readings) if x not in self.possibleReadings]
        
        if new_possible_readings:
            # Usa insert() para manter a ordem de prioridade exata da sua lógica original
            for idx, plate in new_possible_readings:
                self.possibleReadings.insert(idx, plate)
            
            # Chama a API se o endpoint estiver configurado, enviando somente as placas novas
            if self.__class__.api_endpoint and self.__class__.use_continuous_tries:
                # Extrai apenas os nomes das placas para o payload
                plates_to_send = [plate for _, plate in new_possible_readings]
                
                payload = {
                    "instance": self.__class__.instance_id,
                    "readings": plates_to_send,
                    "created": self.start_time.isoformat(),
                    "capture_id": self.id
                }
                # Cria e inicia a thread para a chamada de API
                api_thread = threading.Thread(target=self._call_api_async, args=(payload,))
                api_thread.start()

    def close(self):
        """
        Finaliza o rastreamento, salva a imagem de captura e registra os dados.
        """
        if self.is_closing:
            return
        self.is_closing = True

        # Tenta uma última atualização antes de fechar por timeout
        if not self.__class__.use_continuous_tries:
             self._update_and_call_api()

        if not self.finalReading and self.possibleReadings:
            self.finalReading = self.possibleReadings[0]

        logging.info("Leituras realizadas: %s", self.readings)
        logging.info("Leituras possíveis: %s", self.possibleReadings)
        
        logging.info(f"Finalizando passagem {self.id} com leitura final: {self.finalReading}")
        
        image_path = self._save_capture_image()
        
        tracking_data = {
            'id': self.id,
            'instance_id': self.__class__.instance_id,
            'finalReading': self.finalReading,
            'image_path': image_path,
            'readings': dict(self.readings),
            'possibleReadings': self.possibleReadings,
        }
        
        if self.__class__.db_manager:
            self.__class__.db_manager.save_tracking(tracking_data)

        self.__class__.save_suspect_detections()
        
        self.__class__.trackings.pop(self.id, None)

    def _save_capture_image(self) -> str | None:
        """
        Escolhe o melhor frame e salva a imagem.
        """
        if not self.frames or self.__class__.captures_save_path is None:
            logging.warning(f"Nenhum frame para salvar ou caminho de capturas não configurado para o tracking {self.id}.")
            return None
        
        try:
            best_frame = self.chooseBestFrame(self.frames)
            if not best_frame:
                return None

            now = datetime.now()
            folder_path = self.__class__.captures_save_path / str(now.year) / f"{now.month:02d}" / f"{now.day:02d}"
            folder_path.mkdir(parents=True, exist_ok=True)
            
            x1, y1, x2, y2 = best_frame['plate_bounding_box']
            input_name = best_frame['input_name']
            
            filename = f"{self.finalReading} {input_name} {x1}-{y1}-{x2}-{y2} {self.id}.jpg"
            final_path = folder_path / filename
            
            cv2.imwrite(str(final_path), best_frame['input_frame'])
            logging.info(f"Captura salva para placa {self.finalReading} em '{final_path}'")
            return str(final_path)

        except (IOError, cv2.error, KeyError, Exception) as e:
            logging.error(f"Erro ao salvar imagem de captura para o tracking {self.id}: {e}")
            return None
    
    def chooseBestFrame(self, frames: list) -> dict | None:
        """
        Seleciona o melhor frame com uma lógica de fallback progressiva.

        1. Tenta encontrar frames com distância de Levenshtein <= 2.
        2. Se não encontrar, tenta com distância <= 3.
        3. Se não encontrar, tenta com distância <= 4.
        4. Se ainda assim não encontrar nenhum, ignora o filtro de texto e
        usa a lista de frames original completa.
        
        Após a filtragem, seleciona o frame com a placa mais próxima ao
        centro da imagem. Isso garante que um frame sempre seja retornado,
        desde que a lista inicial não esteja vazia.
        """
        if not frames:
            return None

        candidate_frames = []
        # Loop de tentativas com distâncias progressivas
        for max_distance in [2, 3, 4]:
            # Filtra os frames para a distância atual
            for frame in frames:
                try:
                    reading_text = frame.get('reading', '')
                    if levenshtein(self.text, reading_text) <= max_distance:
                        candidate_frames.append(frame)
                except Exception as e:
                    logging.warning(f"Erro ao processar o texto do frame na filtragem (distância {max_distance}): {e}")
                    continue
            
            # Se encontramos candidatos, paramos de procurar e usamos essa lista
            if candidate_frames:
                # logging.info(f"Encontrados {len(candidate_frames)} frames candidatos com distância Levenshtein <= {max_distance}.")
                break

        # --- FALLBACK FINAL ---
        # Se, após todas as tentativas, a lista de candidatos ainda estiver vazia,
        # usamos a lista original de frames como candidata.
        # O importante é não ficar sem salvar.
        if not candidate_frames:
            # logging.warning("Nenhum frame encontrado com Levenshtein <= 4. Usando todos os frames para critério de centralização.")
            candidate_frames = frames

        # Agora, aplicamos a lógica de encontrar o mais centralizado
        # na lista de candidatos que foi definida (seja por Levenshtein ou pelo fallback)
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
                logging.warning(f"Frame candidato malformado ao calcular distância: {e}")
                continue

        return best_frame

    def definePossibleReadings(self, plates: dict) -> list:
        """
        Algoritmo para pontuar e classificar as leituras de placas.
        """
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
    
    # --- Métodos de Classe ---

    def convert_readings_to_formats(self, formats: List[str], plates: List[str], conv: Dict[str, List[str]]) -> Dict[str, List[str]]:
        # Este método permanece inalterado
        def is_letter(c: str) -> bool: return c.isalpha()
        def is_digit(c: str) -> bool: return c.isdigit()
        def fits_token(ch: str, token: str) -> bool:
            if token == 'L': return is_letter(ch)
            if token == 'N': return is_digit(ch)
            raise ValueError(f"Token inválido: {token}")
        formats_by_len: Dict[int, List[str]] = {}
        for f in formats: formats_by_len.setdefault(len(f), []).append(f)
        resultados: Dict[str, List[str]] = {}
        for plate in plates:
            fmt_list = formats_by_len.get(len(plate), [])
            if not fmt_list: continue
            variantes_set = set()
            for fmt in fmt_list:
                mismatch_positions, candidate_lists = [], []
                ok_entire = True
                for i, tok in enumerate(fmt):
                    ch = plate[i]
                    if fits_token(ch, tok): continue
                    ok_entire = False
                    cand = conv.get(ch)
                    if not cand: candidate_lists = []; break
                    cand_ok = [c for c in cand if fits_token(c, tok)]
                    if not cand_ok: candidate_lists = []; break
                    mismatch_positions.append(i)
                    candidate_lists.append(cand_ok)
                if ok_entire: continue
                if not candidate_lists: continue
                base = list(plate)
                for combo in product(*candidate_lists):
                    buff = base[:]
                    for pos, val in zip(mismatch_positions, combo): buff[pos] = val
                    variantes_set.add(''.join(buff))
            if variantes_set: resultados[plate] = list(variantes_set)
        return resultados
    
    @classmethod
    def setup(cls, db_manager, instance_id: str, captures_save_path: str, 
              suspect_detections_save_path: str, use_continuous_tries: bool = False, 
              reading_formats: list = None, char_corrections: dict = None,
              api_endpoint: str = None):
        cls.db_manager = db_manager
        cls.instance_id = instance_id
        cls.captures_save_path = Path(captures_save_path) if captures_save_path else None
        cls.suspect_detections_save_path = Path(suspect_detections_save_path) if suspect_detections_save_path else None
        cls.use_continuous_tries = use_continuous_tries
        cls.reading_formats = reading_formats
        cls.char_corrections = char_corrections or {}
        cls.api_endpoint = api_endpoint
        
        if cls.captures_save_path: cls.captures_save_path.mkdir(parents=True, exist_ok=True)
        if cls.suspect_detections_save_path: cls.suspect_detections_save_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def newFrame(cls):
        """
        Método chamado a cada novo frame para gerenciar os trackings.
        """
        if not cls.trackings:
            return

        for track in list(cls.trackings.values()):
            track.noFrameCount += 1

            if track.noFrameCount > 5 and not track.is_closing:
                logging.info(f"A passagem {track.id} excedeu o limite de frames sem detecção.")
                
                duration = datetime.now() - track.start_time
                logging.info(f"A captura (timeout) levou {duration.total_seconds():.2f} segundos.")
                
                track.close()

    @classmethod
    def save_suspect_detections(cls):
        # Este método permanece inalterado
        if not cls.suspect_detections: return
        if not cls.suspect_detections_save_path:
            logging.warning("Caminho para salvar detecções suspeitas não configurado.")
            cls.suspect_detections = []
            return
        logging.info(f"Salvando {len(cls.suspect_detections)} detecções suspeitas.")
        for detection in cls.suspect_detections:
            try:
                frame_id = detection["frame_id"]
                frame = detection["frame"]
                x1, y1, x2, y2 = detection["coords"]
                tipo = detection["type"]
                input_name = detection["input_name"]
                filename = f"{input_name} {x1}-{y1}-{x2}-{y2} {tipo} {frame_id}.jpg"
                filepath = cls.suspect_detections_save_path / filename
                cv2.imwrite(str(filepath), frame)
            except (KeyError, cv2.error, Exception) as e:
                logging.error(f"Não foi possível salvar detecção suspeita: {e}")
                continue
        cls.suspect_detections = []