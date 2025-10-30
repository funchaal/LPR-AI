# -*- coding: utf-8 -*-
"""
Módulo de Rastreamento de Passagens.

Este módulo não realiza o rastreamento espacial (frame a frame) de um objeto,
mas sim gerencia a "passagem" ou "sessão" de um veículo detectado. Ele agrupa
múltiplas leituras de placa de um mesmo evento, determina a melhor leitura final,
e orquestra a comunicação com uma API externa e o salvamento dos dados.

A classe `Tracking` atua como um gerenciador (com métodos e estado de classe)
e também como a representação de uma passagem individual (através de suas instâncias).
"""

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import logging
import uuid
import cv2
import requests 
import threading
import time
import json
from typing import Dict, List

from modules.postprocess import RegexValidator, definePossibleReadings, chooseBestFrame, FormatConverter

class Tracking:
    """
    Gerencia e representa a passagem de um veículo, consolidando leituras de placa.
    """
    
    # --- Variáveis de Classe (Estado Global do Gerenciador) ---
    # Este dicionário armazena todas as instâncias de passagens ativas.
    trackings: Dict[str, 'Tracking'] = {}
    
    # Configurações e objetos gerenciadores injetados via setup().
    db_manager = None
    instance_id = None
    captures_save_path = None
    suspect_detections_save_path = None
    use_continuous_tries = False
    skip_same_consecutive_reading = False
    skip_same_consecutive_reading_timeout = 0
    api_endpoint = None
    auth = None
    max_no_frame_count = 10
    reading_filter_by_regex = None
    format_converter = None
    logger = None

    last_possible_readings = []
    last_possible_readings_time = None

    # --- Métodos de Instância (para cada passagem individual) ---

    def __init__(self, logger: logging.Logger):
        """Inicializa uma nova instância de passagem de veículo."""
        self.logger = logger
        self.id = str(uuid.uuid4())         # ID único para esta passagem.
        self.start_time = datetime.now()   # Momento em que a primeira detecção ocorreu.
        self.readings = defaultdict(int)   # Dicionário para contar a frequência de cada leitura de placa.
        self.possibleReadings = []         # Lista de leituras válidas após filtros.
        self.finalReading = ''              # A leitura final decidida (pela API ou localmente).
        self.noFrameCount = 0              # Contador de frames sem detecção (para timeout).
        self.frames = []                   # Armazena os dados dos frames capturados para esta passagem.
        self.closed = False                # Flag: True se a passagem foi finalizada e salva.
        self.closing = False               # Flag: True se o processo de finalização foi iniciado.
        self.leftTheFrame = False          # Flag: True se o objeto saiu do quadro (timeout).
        self.had_same_consecutive_possible_reading = False # Flag: True se já teve a mesma leitura possível que a passagem anterior.
        self.api_calls = 0                 # Contador de chamadas de API pendentes.
        self.api_returned_200 = False      # Flag: True se a API já confirmou uma leitura.
        self._lock = threading.Lock()      # Lock para garantir a segurança em operações concorrentes.

    def __str__(self) -> str:
        return f'ID: {self.id}, Placas Possíveis: {self.possibleReadings}, Placa Final: {self.finalReading}'

    def addCapture(self, capture_data: dict):
        """Adiciona uma nova detecção (leitura e frame) a esta passagem."""
        with self._lock:
            if self.closed or self.closing:
                return # Ignora se a passagem já está sendo finalizada.
                
            self.readings[capture_data['reading']] += 1
            self.frames.append(capture_data)
            self.noFrameCount = 0 # Reseta o contador de timeout, pois o objeto foi visto.

            # Se o modo de tentativas contínuas estiver ativo, processa e chama a API imediatamente.
            if self.__class__.use_continuous_tries:
                self._update_and_call_api()

    def setPossibleReadings(self):
        if self.__class__.format_converter:
            original_readings = list(self.readings.keys())
            converted_readings_dict = self.__class__.format_converter.convert(original_readings)
            updated_readings = self.readings.copy()
            for plate, converted in converted_readings_dict.items():
                score = self.readings.get(plate, 0)
                for converted_plate in converted:
                    updated_readings[converted_plate] = updated_readings.get(converted_plate, 0) + score
            self.readings = updated_readings

        # Define as leituras mais prováveis e aplica um filtro regex.
        possible = definePossibleReadings(self.readings)
        filtered_possible = self.__class__.reading_filter_by_regex.filter_list(possible)

        # Determina quais placas novas devem ser enviadas para a API.
        new_possible_readings = [p for p in filtered_possible if p not in self.possibleReadings]

        if self.__class__.skip_same_consecutive_reading and self.__class__.last_possible_readings:
            time_since_last = (
                (self.start_time - self.__class__.last_possible_readings_time).total_seconds()
                if self.__class__.last_possible_readings_time else None
            )

            # Verifica se há interseção entre as listas
            has_common_reading = any(
                r in self.__class__.last_possible_readings for r in new_possible_readings
            )

            if has_common_reading and (
                time_since_last is None
                or time_since_last < self.__class__.skip_same_consecutive_reading_timeout
                or not self.__class__.skip_same_consecutive_reading_timeout
            ):
                self.logger.info(
                    f"Ignorando leitura repetida '{self.__class__.last_final_reading}' para passagem {self.id}."
                )
                
                # <<< LINHA CORRIGIDA AQUI >>>
                # Remove de new_possible_readings todos os itens que também estão na lista de leituras anteriores.
                new_possible_readings = [
                    r for r in new_possible_readings 
                    if r not in self.__class__.last_possible_readings
                ]
                
                self.had_same_consecutive_possible_reading = True

        if new_possible_readings:
            self.logger.info(f"Novas leituras possíveis para passagem {self.id}: {new_possible_readings}")
            self.possibleReadings.extend(new_possible_readings)

        return new_possible_readings

    def _call_api_async(self, payload: dict):
        """Envia os dados para a API externa em uma thread separada para não bloquear."""
        if self.api_returned_200:
            return # Se a API já deu sucesso, não envia novamente.

        try:
            self.api_calls += 1
            self.logger.info(f"[API] Enviando dados para passagem {self.id}: {payload['readings']}")

            response = requests.post(self.__class__.api_endpoint, json=payload, timeout=30, auth=self.__class__.auth)

            # Verifica se a passagem ainda existe (pode ter sido removida por timeout).
            if not self.__class__.trackings.get(self.id):
                self.logger.warning(f"[API] Resposta recebida para passagem {self.id}, mas ela já foi removida.")
                return
    
            if response.status_code == 200:
                try:
                    data = response.json()
                    api_status = data.get('status')
                    content = data.get('content')

                    if api_status == 200: # Sucesso, a API encontrou a placa.
                        if self.api_returned_200: return # Evita processamento duplicado.
                        
                        self.api_returned_200 = True
                        self.api_calls = 0
                        correct_reading = content
                        self.logger.info(f"[API] Sucesso para passagem {self.id}. Leitura correta: {correct_reading}")
                        
                        with self._lock:
                            self.finalReading = correct_reading
                            # Inicia o fechamento se estiver no modo contínuo.
                            if self.__class__.use_continuous_tries and not self.closing:
                                self._start_async_close()

                    elif api_status == 204: # A API não encontrou a placa na base de dados.
                        if self.api_returned_200: return
                        self.api_calls -= 1 
                        self.logger.info(f"[API] Placa não encontrada (204) para passagem {self.id}. Tentando próximas leituras.")
                    
                    else: # Status inesperado dentro do JSON.
                        self.logger.warning(f"[API] Status interno inesperado ({api_status}) para {self.id}. Resposta: {content}")

                except json.JSONDecodeError:
                    self.logger.error(f"[API] Falha ao decodificar JSON para {self.id}. Resposta: {response.text}")
            else:
                self.logger.error(f"[API] Erro de comunicação para {self.id}. Status: {response.status_code}, Resposta: {response.text}")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"[API] Falha na chamada para {self.id}: {e}")
            if self.api_calls > 0: self.api_calls -= 1

    def _update_and_call_api(self):
        """Filtra, formata e envia as leituras para a API se houver novas candidatas."""
        # Converte formatos de placa (ex: Mercosul para Padrão) para aumentar chances de match.
        new_possible_readings = self.setPossibleReadings()

        # Se não houver API, define a melhor leitura local e fecha a passagem.
        if not self.__class__.api_endpoint and self.possibleReadings:
            self.finalReading = self.possibleReadings[0]
            self._start_async_close()
        # Se houver API e novas placas, envia.
        elif new_possible_readings and self.__class__.api_endpoint:
            payload = {
                "instance": self.__class__.instance_id,
                "readings": new_possible_readings,
                "created": self.start_time.isoformat(),
                "capture_id": self.id
            }
            threading.Thread(target=self._call_api_async, args=(payload,), daemon=True).start()

    def _start_async_close(self):
        """Inicia o processo de fechamento em uma thread separada para não bloquear."""
        if self.closed or self.closing: return
        self.closing = True
        self.logger.info(f"Iniciando fechamento assíncrono da passagem {self.id}...")
        threading.Thread(target=self._async_close_worker, daemon=True).start()

    def _async_close_worker(self):
        """Worker que executa o fechamento em background."""
        try:
            # Aguarda a finalização de chamadas pendentes da API (com timeout).
            if self.api_calls > 0:
                self.logger.info(f"Aguardando retorno da API para {self.id} (máx 30s)... ({self.api_calls} chamadas pendentes)")
                timeout_start = time.time()
                while self.api_calls > 0 and (time.time() - timeout_start) < 30 and not self.api_returned_200:
                    time.sleep(0.1)
                if self.api_calls > 0 and not self.api_returned_200:
                    self.logger.warning(f"Timeout aguardando API para {self.id}. Prosseguindo com fechamento.")

            # Chama um endpoint de "fechamento" se configurado.
            if self.__class__.close_api_endpoint:
                self._call_close_api()

            self._execute_final_close()
        except Exception as e:
            self.logger.error(f"Erro durante fechamento assíncrono de {self.id}: {e}", exc_info=True)
            self._cleanup_tracking() # Garante a limpeza em caso de erro.

    def _call_close_api(self):
        """Chama o endpoint secundário de finalização de evento."""
        try:
            payload = {"instance": self.__class__.instance_id, "capture_id": self.id, "final_reading": self.finalReading}
            self.logger.info(f"[API-Close] Enviando dados de fechamento para {self.id}")
            response = requests.post(self.__class__.close_api_endpoint, json=payload, timeout=30, auth=self.__class__.auth)
            if response.status_code != 200:
                self.logger.error(f"[API-Close] Erro para {self.id}. Status: {response.status_code}, Resposta: {response.text}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"[API-Close] Falha ao chamar para {self.id}: {e}")

    def _execute_final_close(self):
        """Executa a lógica final de fechamento: define a leitura, salva e limpa."""
        with self._lock:
            if self.closed: return
            self.closed = True

        is_suspect = False
        # Se a API não definiu uma leitura final, usa a melhor leitura local.
        if not self.finalReading and self.readings:
            # Recalcula as leituras possíveis se ainda não foram definidas.
            if not self.possibleReadings:
                self._update_and_call_api() # Re-executa a lógica de definição de leituras

            if self.possibleReadings:
                if not self.finalReading:
                    self.finalReading = self.possibleReadings[int(len(self.possibleReadings) / 2)]
            else:
                self.finalReading = '...' # Marca como suspeito se nenhuma leitura válida foi encontrada.
                is_suspect = True

        if not is_suspect:
            self.logger.info(f"Finalizando passagem {self.id} | Duração: {(datetime.now() - self.start_time).total_seconds():.2f}s | Leitura Final: {self.finalReading}")
            self.logger.debug(f"Leituras realizadas para a passagem {self.id}: {self.readings}")
            self.logger.debug(f"Leituras possíveis para a passagem {self.id}: {self.possibleReadings}")

            self.__class__.last_possible_readings = self.possibleReadings.copy()

            image_path = self._save_capture_image()
            if self.__class__.db_manager:
                try:
                    tracking_data = {
                        'id': self.id, 'instance_id': self.__class__.instance_id, 'finalReading': self.finalReading,
                        'image_path': image_path, 'readings': dict(self.readings), 'possibleReadings': self.possibleReadings,
                    }
                    self.__class__.db_manager.save_tracking(tracking_data)
                except Exception as e:
                    self.logger.error(f"Erro ao salvar passagem {self.id} no banco: {e}")
        else:
            if self.had_same_consecutive_possible_reading:
                self.logger.info(f"Passagem {self.id} desconsiderada devido a leitura repetida igual a passagem anterior.")
            else:
                self.logger.warning(f"Passagem {self.id} marcada como suspeita.")
                self._save_capture_image(is_suspect=True)
        
        # Limpa da memória se for suspeito ou se o objeto já saiu do quadro.
        if is_suspect or self.leftTheFrame:
            self._cleanup_tracking()

    def _cleanup_tracking(self):
        """Remove a instância da passagem do dicionário de rastreamentos ativos."""
        if self.__class__.trackings.pop(self.id, None):
            self.logger.info(f"Passagem {self.id} removida da memória.")

    def _save_capture_image(self, is_suspect=False) -> str | None:
        if not self.__class__.save_captures:
            return None
        """Escolhe o melhor frame da passagem, salva como imagem e retorna o caminho."""
        if not self.frames: return None
        save_dir = self.__class__.suspect_detections_save_path if is_suspect else self.__class__.captures_save_path
        if not save_dir: return None
        
        try:
            best_frame = chooseBestFrame(self.frames, self.logger, self.finalReading)
            if not best_frame: return None

            now = datetime.now()
            folder_path = save_dir / str(now.year) / f"{now.month:02d}" / f"{now.day:02d}"
            folder_path.mkdir(parents=True, exist_ok=True)
            
            x1, y1, x2, y2 = best_frame['bounding_box']
            filename_plate = self.finalReading
            filename = f"{filename_plate} {self.instance_id} {best_frame['input_name']} {x1}-{y1}-{x2}-{y2} {self.id}.jpg"
            final_path = folder_path / filename
            
            cv2.imwrite(str(final_path), best_frame['input_frame'])
            self.logger.info(f"Captura salva para {self.id} em '{final_path}'")
            return str(final_path)
        except Exception as e:
            self.logger.error(f"Erro ao salvar imagem para {self.id}: {e}", exc_info=True)
            return None
    
    # --- Métodos de Classe (Gerenciador) ---
    
    @classmethod
    def setup(cls, db_manager, instance_id: str, save_captures: bool, captures_save_path: str, save_suspect_detections: bool,
              suspect_detections_save_path: str, use_continuous_tries: bool, skip_same_consecutive_reading: bool, skip_same_consecutive_reading_timeout: int, reading_formats: list,
              readings_filter_regex: str, char_corrections: dict, max_no_frame_count: int, 
              api_endpoint: str, api_user: str, api_password: str, close_api_endpoint: str, logger: logging.Logger):
        """Configura as variáveis de classe (estado global) do gerenciador de rastreamento."""
        cls.db_manager = db_manager
        cls.instance_id = instance_id
        cls.save_captures = save_captures
        cls.captures_save_path = Path(captures_save_path) if captures_save_path else None
        cls.save_suspect_detections = save_suspect_detections
        cls.suspect_detections_save_path = Path(suspect_detections_save_path) if suspect_detections_save_path else None
        cls.use_continuous_tries = use_continuous_tries
        cls.skip_same_consecutive_reading = skip_same_consecutive_reading
        cls.skip_same_consecutive_reading_timeout = skip_same_consecutive_reading_timeout
        cls.max_no_frame_count = max_no_frame_count
        cls.api_endpoint = api_endpoint
        cls.close_api_endpoint = close_api_endpoint
        cls.auth = (api_user, api_password) if api_user and api_password else None

        cls.reading_filter_by_regex = RegexValidator(readings_filter_regex, logger)
        cls.format_converter = FormatConverter(reading_formats, char_corrections or {}) if reading_formats else None
        
        cls.logger = logger

        if cls.captures_save_path: cls.captures_save_path.mkdir(parents=True, exist_ok=True)
        if cls.suspect_detections_save_path: cls.suspect_detections_save_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def newFrame(cls):
        """
        Avança o estado de todas as passagens ativas. Chamado a cada novo frame do vídeo.
        Este método atua como o "coração" do gerenciador, lidando com timeouts.
        """
        if not cls.trackings: return

        for track in list(cls.trackings.values()):
            track.noFrameCount += 1

            if track.noFrameCount > 1:
                logging.debug(f"Passagem {track.id} sem detecção por {track.noFrameCount} frames.")

            # Verifica se a passagem excedeu o limite de frames sem detecção.
            if track.noFrameCount > cls.max_no_frame_count:
                # Apenas loga na primeira vez que o timeout é atingido.
                if not track.leftTheFrame:
                    cls.logger.info(f"Passagem {track.id} excedeu limite de frames sem detecção. Fechando...")
                    track.leftTheFrame = True
                
                # Inicia o fechamento se ainda não estiver em progresso.
                if not track.closing:
                    track._start_async_close()
                # Se já estiver fechando e não houver mais chamadas de API, força a limpeza.
                elif track.api_calls == 0:
                    track._cleanup_tracking()