# tracking.py

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import logging
import uuid
import cv2

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
    api = None

    # --- Métodos de Instância (para cada passagem individual) ---

    def __init__(self):
        """Inicializa um novo objeto de rastreamento."""
        self.start_time = datetime.now()
        self.id = str(uuid.uuid4())
        self.readings = defaultdict(int)
        self.possibleReadings = []
        self.finalReading = ''
        self.noFrameCount = 0
        self.frames = []  # Armazena os dados dos frames relevantes

    def __str__(self):
        """Representação em string do objeto de rastreamento."""
        return f'ID: {self.id}, Placas Possíveis: {self.possibleReadings}, Placa Final: {self.finalReading}'

    def addCapture(self, reading: str, frame_data: dict):
        """
        Adiciona uma nova leitura e os dados do frame a este rastreamento.

        Args:
            reading (str): A leitura da placa identificada.
            frame_data (dict): Dicionário contendo o frame e metadados.
        """
        self.readings[reading] += 1
        self.frames.append(frame_data)
        self.noFrameCount = 0

        if self.__class__.use_continuous_tries:
            self._update_and_call_api()

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
                    # Add the converted plate with the original score, summing if it already exists
                    updated_readings[converted_plate] = updated_readings.get(converted_plate, 0) + original_score
            
            self.readings = updated_readings

        defined_possible_readings = self.definePossibleReadings(self.readings)
        new_possible_readings = [[idx, x] for idx, x in enumerate(defined_possible_readings) if x not in self.possibleReadings]
        
        if new_possible_readings:
            for idx, plate in new_possible_readings:
                self.possibleReadings.insert(idx, plate)
            
            if self.__class__.api:
                self.__class__.api(instance=self.__class__.instance_id, readings=[plate for _, plate in new_possible_readings])

    def close(self):
        """
        Finaliza o rastreamento, salva a imagem de captura e registra os dados
        através do gerenciador de banco de dados.
        """
        logging.info(f"Finalizando passagem {self.id} com leitura final: {self.finalReading}")
        
        # 1. Salva a imagem da melhor captura e obtém seu caminho
        image_path = self._save_capture_image()
        
        # 2. Prepara um dicionário com todos os dados a serem salvos
        tracking_data = {
            'id': self.id,
            'instance_id': self.__class__.instance_id,
            'finalReading': self.finalReading,
            'image_path': image_path,
            'readings': dict(self.readings), # Converte defaultdict para dict para serialização
            'possibleReadings': self.possibleReadings,
        }
        
        # 3. Usa o gerenciador de banco de dados para salvar os dados
        if self.__class__.db_manager:
            self.__class__.db_manager.save_tracking(tracking_data)
        
        # 4. Remove o rastreamento do dicionário ativo
        del self.__class__.trackings[self.id]

    def _save_capture_image(self) -> str | None:
        """
        Escolhe o melhor frame, salva a imagem em uma estrutura de pastas
        organizada por data e retorna o caminho completo do arquivo.
        """
        if not self.frames or self.__class__.captures_save_path is None:
            logging.warning(f"Nenhum frame para salvar ou caminho de capturas não configurado para o tracking {self.id}.")
            return None
        
        try:
            best_frame = self.chooseBestFrame(self.frames)
            if not best_frame:
                return None

            now = datetime.now()
            # Uso de pathlib para manipulação segura e limpa de caminhos
            folder_path = self.__class__.captures_save_path / str(now.year) / f"{now.month:02d}" / f"{now.day:02d}"
            folder_path.mkdir(parents=True, exist_ok=True)
            
            x1, y1, x2, y2 = best_frame['plate_bounding_box']
            input_name = best_frame['input_name']
            
            filename = f"{self.finalReading} {input_name} {x1}-{y1}-{x2}-{y2} {self.id}.jpg"
            final_path = folder_path / filename
            
            # Salva a imagem no disco
            cv2.imwrite(str(final_path), best_frame['input_frame'])
            logging.info(f"Captura salva para placa {self.finalReading} em '{final_path}'")
            return str(final_path)

        except (IOError, cv2.error, KeyError, Exception) as e:
            logging.error(f"Erro ao salvar imagem de captura para o tracking {self.id}: {e}")
            return None

    def chooseBestFrame(self, frames: list) -> dict | None:
        """
        Seleciona o melhor frame da lista com base na proximidade da
        bounding box da placa ao centro da imagem.
        """
        if not frames:
            return None

        best_frame = None
        min_distance = float('inf')

        for frame in frames:
            try:
                height, width = frame['input_frame'].shape[:2]
                image_center_x = width / 2
                image_center_y = height / 2

                x1, y1, x2, y2 = frame['plate_bounding_box']
                bbox_center_x = (x1 + x2) / 2
                bbox_center_y = (y1 + y2) / 2
                
                # Fórmula da distância euclidiana
                distance = ((bbox_center_x - image_center_x) ** 2 + (bbox_center_y - image_center_y) ** 2) ** 0.5

                if distance < min_distance:
                    min_distance = distance
                    best_frame = frame
            except (KeyError, TypeError) as e:
                logging.warning(f"Frame malformado ou faltando chaves ao escolher o melhor frame: {e}")
                continue

        return best_frame

    def definePossibleReadings(self, plates: dict) -> list:
        """
        Algoritmo para pontuar e classificar as leituras de placas com base em
        substrings comuns, retornando as mais prováveis.
        """
        plate_pontuation = defaultdict(int)

        for plate in plates.keys():
            # Gera substrings de tamanho 2 até o tamanho da própria placa
            substrings = {
                plate[j:j + i]
                for i in range(2, len(plate) + 1)
                for j in range(len(plate) - i + 1)
            }
            # Compara com todas as outras leituras
            for substring in substrings:
                for reading, count in plates.items():
                    if substring in reading:
                        plate_pontuation[plate] += plates[plate]

        # Retorna as 2 placas com maior pontuação
        top_plates = sorted(plate_pontuation, key=plate_pontuation.get, reverse=True)[:2]
        return top_plates
    
    # --- Métodos de Classe (para gerenciar todos os trackings) ---

    def convert_readings_to_formats(
        self,
        formats: List[str],
        plates: List[str],
        conv: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Retorna {placa_original: [possibilidades_convertidas]} SEM prioridade de ordem.

        Regras:
        - Se a placa encaixa em UM formato, ainda assim tentamos propor conversões
        para os OUTROS formatos (apenas onde houver mismatch e conversão possível).
        - Se não houver nenhum formato do mesmo tamanho, nada é retornado para aquela placa.
        - Usa set para deduplicação com custo O(1) por inserção (ordem não garantida).
        """

        def is_letter(c: str) -> bool:
            return c.isalpha()

        def is_digit(c: str) -> bool:
            return c.isdigit()

        def fits_token(ch: str, token: str) -> bool:
            if token == 'L':
                return is_letter(ch)
            if token == 'N':
                return is_digit(ch)
            raise ValueError(f"Token inválido: {token}")

        # Agrupa formatos por tamanho
        formats_by_len: Dict[int, List[str]] = {}
        for f in formats:
            formats_by_len.setdefault(len(f), []).append(f)

        resultados: Dict[str, List[str]] = {}

        for plate in plates:
            fmt_list = formats_by_len.get(len(plate), [])
            if not fmt_list:
                continue  # nenhum formato compatível em tamanho

            variantes_set = set()

            for fmt in fmt_list:
                mismatch_positions = []
                candidate_lists = []

                ok_entire = True
                for i, tok in enumerate(fmt):
                    ch = plate[i]
                    if fits_token(ch, tok):
                        continue
                    ok_entire = False
                    cand = conv.get(ch)
                    if not cand:
                        candidate_lists = []
                        break
                    cand_ok = [c for c in cand if fits_token(c, tok)]
                    if not cand_ok:
                        candidate_lists = []
                        break
                    mismatch_positions.append(i)
                    candidate_lists.append(cand_ok)

                # Se já encaixa nesse formato específico, não precisamos propor conversão para ele
                if ok_entire:
                    continue

                # Se houve mismatch mas não há candidatos viáveis, pula este formato
                if not candidate_lists:
                    continue

                # Gera combinações de substituições
                base = list(plate)
                for combo in product(*candidate_lists):
                    buff = base[:]
                    for pos, val in zip(mismatch_positions, combo):
                        buff[pos] = val
                    variantes_set.add(''.join(buff))

            if variantes_set:
                resultados[plate] = list(variantes_set)

        return resultados

    @classmethod
    def setup(cls, db_manager, instance_id: str, captures_save_path: str, 
              suspect_detections_save_path: str, use_continuous_tries: bool = False, reading_formats: list = None, char_corrections: dict = None,
              api=None):
        """
        Configura as variáveis de classe e dependências.

        Args:
            db_manager: Instância de um gerenciador de banco de dados.
            instance_id (str): Identificador desta instância do programa.
            captures_save_path (str): Caminho da pasta base para salvar capturas.
            suspect_detections_save_path (str): Caminho para salvar detecções suspeitas.
            use_continuous_tries (bool): Flag para chamar a API continuamente.
            api (callable, optional): Função da API externa.
        """
        cls.db_manager = db_manager
        cls.instance_id = instance_id
        # Converte strings de caminho para objetos Path
        cls.captures_save_path = Path(captures_save_path) if captures_save_path else None
        cls.suspect_detections_save_path = Path(suspect_detections_save_path) if suspect_detections_save_path else None
        cls.use_continuous_tries = use_continuous_tries
        cls.reading_formats = reading_formats
        cls.char_corrections = char_corrections or {}
        cls.api = api or (lambda instance, readings: readings[0] if readings else None)
        
        # Garante que os diretórios base existam
        if cls.captures_save_path:
            cls.captures_save_path.mkdir(parents=True, exist_ok=True)
        if cls.suspect_detections_save_path:
            cls.suspect_detections_save_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def newFrame(cls):
        """
        Método a ser chamado a cada novo frame do vídeo. Incrementa o contador
        de 'sem frames' e fecha os trackings que expiraram.
        """
        if not cls.trackings:
            return

        # Itera sobre uma cópia da lista de valores para poder modificar o dicionário
        for track in list(cls.trackings.values()):
            track.noFrameCount += 1

            # Se um objeto não é visto por mais de 5 frames, ele é finalizado
            if track.noFrameCount > 5:
                logging.info(f"A passagem {track.id} excedeu o limite de frames sem detecção.")
                
                # Tenta uma última vez definir as placas possíveis
                track._update_and_call_api()
                
                # Define a leitura final com base na placa mais provável
                if track.possibleReadings:
                    track.finalReading = track.possibleReadings[0]

                duration = datetime.now() - track.start_time
                logging.info(f"A captura da placa {track.finalReading} levou {duration.total_seconds():.2f} segundos.")
                
                track.close()

    @classmethod
    def save_suspect_detections(cls):
        """
        Salva imagens recortadas de detecções consideradas suspeitas.
        """
        if not cls.suspect_detections:
            return
        
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

                cropped_image = frame[y1:y2, x1:x2]
                
                filename = f"{input_name} {x1}-{y1}-{x2}-{y2} {tipo} {frame_id}.jpg"
                filepath = cls.suspect_detections_save_path / filename

                cv2.imwrite(str(filepath), cropped_image)
            except (KeyError, cv2.error, Exception) as e:
                logging.error(f"Não foi possível salvar detecção suspeita: {e}")
                continue

        cls.suspect_detections = [] # Limpa a lista após salvar