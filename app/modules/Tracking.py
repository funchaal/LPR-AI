# tracking.py

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import logging
import uuid
import cv2
import requests 
import threading
import time

from itertools import product
from typing import Dict, List

from modules.postprocess import RegexValidator, definePossibleReadings, chooseBestFrame, FormatConverter

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
        """
        Inicializa um novo objeto de rastreamento.
        """
        self.start_time = datetime.now()
        self.id = str(uuid.uuid4())
        self.readings = defaultdict(int)
        self.possibleReadings = []
        self.finalReading = ''
        self.noFrameCount = 0
        self.frames = []
        self.closed = False
        self.closing = False  # Flag para indicar que está sendo fechado
        self.api_calls = 0
        self.api_returned_200 = False
        self.leftTheFrame = False
        self._lock = threading.Lock()

    def __str__(self):
        """
        Representação em string do objeto de rastreamento.
        """
        return f'ID: {self.id}, Placas Possíveis: {self.possibleReadings}, Placa Final: {self.finalReading}'

    def addCapture(self, capture_data: dict):
        """
        Adiciona uma nova leitura e os dados do frame a este rastreamento.
        """
        with self._lock:
            if self.closed or self.closing:
                return
                
            self.readings[capture_data['reading']] += 1
            self.frames.append(capture_data)
            self.noFrameCount = 0

            if self.__class__.use_continuous_tries:
                self._update_and_call_api()

    def _call_api_async(self, payload: dict):
        """
        Função alvo da thread para chamar a API de forma assíncrona.
        """
        if self.api_returned_200:
            return

        try:
            self.api_calls += 1
            logging.info(f"Enviando dados para API para captura {self.id}: {payload['readings']}")

            response = requests.post(self.__class__.api_endpoint, json=payload, timeout=30, auth=self.__class__.auth)

            # Verifica se o tracking ainda existe (pode ter sido removido)
            track_instance = self.__class__.trackings.get(self.id)
            if not track_instance:
                logging.warning(f"API retornou para a captura {self.id}, mas ela já foi removida.")
                return
    
            if response.status_code == 200:
                if self.api_returned_200:
                    logging.info(f"API já retornou 200 anteriormente para captura {self.id}. Ignorando resposta duplicada.")
                    return
                self.api_returned_200 = True
                self.api_calls = 0

                correct_reading = response.text.strip()
                logging.info(f"API retornou 200 para captura {self.id}. Leitura correta: {correct_reading}")
                
                with track_instance._lock:
                    track_instance.finalReading = correct_reading

                    if self.__class__.use_continuous_tries and not track_instance.closing:
                        track_instance._start_async_close()

            elif response.status_code == 204:
                if self.api_returned_200:
                    logging.info(f"API já retornou 200 anteriormente para captura {self.id}. Ignorando resposta duplicada.")
                    return
                self.api_calls -= 1 
                logging.info(f"API retornou 204 para captura {self.id}. Nenhuma leitura correspondeu. Continuando...")
            
            else:
                logging.error(f"Erro na API para captura {self.id}. Status: {response.status_code}, Resposta: {response.text}")

        except requests.exceptions.RequestException as e:
            logging.error(f"Falha ao chamar a API para captura {self.id}: {e}")

    def _update_and_call_api(self):
        """
        Verifica novas placas possíveis e chama a API externa, se configurado.
        """
        if self.__class__.format_converter is not None:
            original_readings = list(self.readings.keys())
            converted_readings_dict = self.__class__.format_converter.convert(original_readings)
            
            updated_readings = self.readings.copy()
            for original_plate, converted_readings in converted_readings_dict.items():
                original_score = self.readings.get(original_plate, 0)
                for converted_plate in converted_readings:
                    updated_readings[converted_plate] = updated_readings.get(converted_plate, 0) + original_score
            self.readings = updated_readings

        defined_possible_readings = definePossibleReadings(self.readings)
        defined_possible_readings = self.__class__.reading_filter_by_regex.filter_list(defined_possible_readings)

        # Para modo final, envia todas as leituras possíveis
        if self.__class__.use_continuous_tries:
            plates_to_send = [plate for plate in defined_possible_readings if plate not in self.possibleReadings]
            if plates_to_send:
                # Atualiza a lista para evitar reenvios
                for plate in plates_to_send:
                    if plate not in self.possibleReadings:
                        self.possibleReadings.append(plate)
        else:
            plates_to_send = defined_possible_readings
            self.possibleReadings = defined_possible_readings

        if not self.__class__.api_endpoint and self.possibleReadings:
            self.finalReading = self.possibleReadings[0]
            self._start_async_close()
        elif plates_to_send and self.__class__.api_endpoint:
            payload = {
                "instance": self.__class__.instance_id,
                "readings": plates_to_send,
                "created": self.start_time.isoformat(),
                "capture_id": self.id
            }

            api_thread = threading.Thread(target=self._call_api_async, args=(payload,))
            api_thread.daemon = True  # Thread daemon para não bloquear o fechamento da aplicação
            api_thread.start()

    def _start_async_close(self):
        """
        Inicia o fechamento em uma thread separada.
        """
        if self.closed or self.closing:
            return
        self.closing = True

        logging.info(f"Iniciando fechamento assíncrono da passagem {self.id}...")

        close_thread = threading.Thread(target=self._async_close_worker)
        close_thread.daemon = True
        close_thread.start()

    def _async_close_worker(self):
        """
        Worker que executa o fechamento em background.
        Aguarda a API se necessário, mas não bloqueia o código principal.
        """
        try:
            # Se não tem uso contínuo e tem API, faz uma última chamada
            if not self.__class__.use_continuous_tries and self.__class__.api_endpoint:
                logging.info(f"Fazendo chamada final à API para {self.id}")
                self._update_and_call_api()

            # Aguarda a API por no máximo 15 segundos se houver pendência
            if self.api_calls > 0:
                logging.info(f"Aguardando retorno da API para {self.id} (máximo 30s)")
                timeout_start = time.time()
                while (self.api_calls > 0 and (time.time() - timeout_start) < 30) and not self.api_returned_200:
                    time.sleep(0.1)

                if self.api_calls > 0:
                    logging.warning(f"Timeout aguardando API para {self.id}. Prosseguindo com fechamento.")

            # Executa o fechamento final
            self._execute_final_close()

        except Exception as e:
            logging.error(f"Erro durante fechamento assíncrono de {self.id}: {e}")
            # Garante que sempre será removido da memória
            self._cleanup_tracking()

    def _execute_final_close(self):
        """
        Executa o fechamento final e salva os dados.
        """
        with self._lock:
            self.closing = True
            self.api_calls = 0
            if self.closed:
                return
            self.closed = True

        is_suspect = False

        # Se a API não definiu finalReading, usa a melhor local
        if not self.finalReading and self.readings:
            if not self.possibleReadings:
                if self.__class__.format_converter is not None:
                    # Obtém as leituras originais
                    original_readings = list(self.readings.keys())
                    
                    # Converte as leituras usando o format_converter da classe
                    converted_readings_dict = self.__class__.format_converter.convert(original_readings)
                    
                    # Cria uma cópia das leituras atuais para atualizar
                    updated_readings = self.readings.copy()
                    
                    # Atualiza as leituras convertidas somando os scores
                    for original_plate, converted_readings in converted_readings_dict.items():
                        original_score = self.readings.get(original_plate, 0)
                        for converted_plate in converted_readings:
                            updated_readings[converted_plate] = updated_readings.get(converted_plate, 0) + original_score

                    self.readings = updated_readings

                # Define as leituras possíveis
                defined_possible_readings = definePossibleReadings(self.readings)
                defined_possible_readings = self.__class__.reading_filter_by_regex.filter_list(defined_possible_readings)

                # Atualiza possibleReadings filtrando novamente (pode ser opcional)
                self.possibleReadings = defined_possible_readings

            if self.possibleReadings:
                self.finalReading = self.possibleReadings[0]
            else:
                self.finalReading = '...'
                is_suspect = True

        if not is_suspect:
            duration = datetime.now() - self.start_time
            logging.info(f"Finalizando passagem {self.id} (duração: {duration.total_seconds():.2f}s)")

            logging.info(f"Leituras realizadas para {self.id}: {dict(self.readings)}")
            logging.info(f"Leituras possíveis para {self.id}: {self.possibleReadings}")
            logging.info(f"Leitura final para {self.id}: {self.finalReading}")
            
            # Salva a imagem
            image_path = self._save_capture_image()
            
            # Prepara dados para salvar no banco
            tracking_data = {
                'id': self.id,
                'instance_id': self.__class__.instance_id,
                'finalReading': self.finalReading,
                'image_path': image_path,
                'readings': dict(self.readings),
                'possibleReadings': self.possibleReadings,
            }
            
            # Salva no banco de dados
            if self.__class__.db_manager:
                try:
                    self.__class__.db_manager.save_tracking(tracking_data)
                    logging.info(f"Dados salvos no banco para {self.id}")
                except Exception as e:
                    logging.error(f"Erro ao salvar no banco para {self.id}: {e}")

        else:
            self._save_capture_image()
        
        # Remove da memória
        if is_suspect or self.leftTheFrame:
            logging.info(f"Passagem {self.id} completamente finalizada e removida da memória.")
            self._cleanup_tracking()

    def _cleanup_tracking(self):
        """
        Remove o tracking das listas de memória.
        """
        logging.info(f"Removendo passagem {self.id} da memória.")
        self.__class__.trackings.pop(self.id, None)

    def _save_capture_image(self, is_suspect=False) -> str | None:
        """
        Escolhe o melhor frame e salva a imagem.
        """
        if not self.frames or self.__class__.captures_save_path is None:
            logging.warning(f"Nenhum frame para salvar ou caminho não configurado para {self.id}")
            return None
        
        try:
            best_frame = chooseBestFrame(self.frames, self.finalReading)
            if not best_frame:
                logging.warning(f"Não foi possível escolher melhor frame para {self.id}")
                return None

            now = datetime.now()
            base_folder_path = None

            if is_suspect and self.__class__.save_suspect_detections and self.__class__.suspect_detections_save_path:
                base_folder_path = self.__class__.suspect_detections_save_path
            else:
                base_folder_path = self.__class__.captures_save_path

            folder_path = base_folder_path / str(now.year) / f"{now.month:02d}" / f"{now.day:02d}"
            folder_path.mkdir(parents=True, exist_ok=True)
            
            x1, y1, x2, y2 = best_frame['bounding_box']
            input_name = best_frame['input_name']
            
            # Garante que a placa final não seja vazia no nome do arquivo
            filename_plate = self.finalReading if self.finalReading else "SEM_LEITURA"
            
            filename = f"{filename_plate} {input_name} {x1}-{y1}-{x2}-{y2} {self.id}.jpg"
            final_path = folder_path / filename
            
            cv2.imwrite(str(final_path), best_frame['input_frame'])
            logging.info(f"Captura salva para {self.id} em '{final_path}'")
            return str(final_path)

        except (IOError, cv2.error, KeyError, Exception) as e:
            logging.error(f"Erro ao salvar imagem para {self.id}: {e}")
            return None
    
    # --- Métodos de Classe ---
    
    @classmethod
    def setup(cls, db_manager, instance_id: str, captures_save_path: str, save_suspect_detections: bool,
              suspect_detections_save_path: str, use_continuous_tries: bool = False, 
              reading_formats: list = None, readings_filter_regex: str = None, char_corrections: dict = None, max_no_frame_count: int = 10, 
              api_endpoint: str = None, api_user: str = None, api_password: str = None):
        """Configura o módulo de Tracking com os parâmetros necessários."""
        cls.db_manager = db_manager
        cls.instance_id = instance_id
        cls.captures_save_path = Path(captures_save_path) if captures_save_path else None
        cls.save_suspect_detections = save_suspect_detections
        cls.suspect_detections_save_path = Path(suspect_detections_save_path) if suspect_detections_save_path else None
        cls.use_continuous_tries = use_continuous_tries

        cls.reading_filter_by_regex = RegexValidator(readings_filter_regex)
        cls.format_converter = FormatConverter(reading_formats, char_corrections or {}) if reading_formats else None

        cls.max_no_frame_count = max_no_frame_count
        
        cls.api_endpoint = api_endpoint
        cls.api_user = api_user
        cls.api_password = api_password

        # Se usuário e senha forem fornecidos, cria o auth
        cls.auth = (api_user, api_password) if api_user and api_password else None

        if cls.captures_save_path: 
            cls.captures_save_path.mkdir(parents=True, exist_ok=True)
        if cls.suspect_detections_save_path: 
            cls.suspect_detections_save_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def newFrame(cls):
        """
        Método chamado a cada novo frame para gerenciar os trackings.
        Para continuous_tries, trackings fechados permanecem até timeout natural.
        """
        if not cls.trackings:
            return

        # Trabalha com trackings ativos
        for track in list(cls.trackings.values()):
            track.noFrameCount += 1
            logging.info(f'NoFrameCount atualmente para a passagem {track.id}: {track.noFrameCount}')

            if track.noFrameCount > cls.max_no_frame_count:
                track.leftTheFrame = True
                logging.info(f"Passagem {track.id} excedeu limite de frames (timeout)")
                
                # Se já está fechando (continuous_tries), apenas move para closing_trackings
                if not track.closing:
                    track._start_async_close()
                elif track.api_calls == 0:
                    track._cleanup_tracking()
