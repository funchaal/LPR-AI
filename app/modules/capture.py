# -*- coding: utf-8 -*-
"""
Módulo de Captura de Vídeo.

Este módulo define a classe `VideoSource`, responsável por abstrair a captura
de frames de diversas fontes de vídeo, como câmeras locais, arquivos de vídeo,
streams RTSP e snapshots HTTP. Inclui funcionalidades para detecção automática
do tipo de fonte, injeção segura de credenciais, e reconexão automática em
caso de falha na comunicação com a fonte.
"""

import cv2
import requests
import numpy as np
import logging
import os
import time
import warnings
from requests.auth import HTTPBasicAuth
from urllib.parse import urlparse, urlunparse, quote

# Define o nível de log do OpenCV como ERROR para suprimir mensagens de informação (ex: backend usado).
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Logger configurado novamente para evitar sobrescrita do logger principal da aplicação por outras bibliotecas.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(processName)s — %(levelname)s — %(message)s"
)

# Ignora avisos específicos do OpenCV sobre dados JPEG corrompidos em streams MJPEG,
# que podem ocorrer frequentemente e não indicam um erro fatal.
warnings.filterwarnings("ignore", category=UserWarning, module="cv2")

class VideoSource:
    """
    Gerencia a captura de frames de diferentes tipos de fontes de vídeo.

    Atributos:
        input_endpoint (str | int): O identificador da fonte (URL, caminho do arquivo ou índice da câmera).
        username (str, opcional): Nome de usuário para autenticação.
        password (str, opcional): Senha para autenticação.
        timeout (int, opcional): Tempo limite em segundos para requisições HTTP.
        cap (cv2.VideoCapture): Objeto de captura do OpenCV para fontes contínuas.
        source_type (str): O tipo de fonte detectado ('camera', 'video', 'rtsp', 'http').
    """

    def __init__(self, input_endpoint, username=None, password=None, timeout=3):
        """Inicializa o objeto VideoSource e determina o tipo de fonte."""
        self.input_endpoint = input_endpoint
        self.username = username
        self.password = password
        self.timeout = timeout
        self.cap = None
        self.source_type = self._detect_source_type()

        # Para fontes de vídeo contínuas, inicializa a captura imediatamente.
        if self.source_type in ["video", "camera", "rtsp"]:
            self._init_capture()

    def _detect_source_type(self) -> str | None:
        """Detecta e retorna o tipo da fonte de vídeo com base no endpoint."""
        if isinstance(self.input_endpoint, int) or str(self.input_endpoint).isdigit():
            return "camera"  # Fonte é um índice de webcam.

        if isinstance(self.input_endpoint, str):
            endpoint = self.input_endpoint.lower()
            
            if endpoint.startswith("rtsp://") or endpoint.endswith(".mjpg"):
                return "rtsp"  # Stream contínuo (RTSP ou MJPEG).
            
            elif endpoint.startswith(("http://", "https://")):
                return "http"  # Snapshot JPG/PNG via requisição HTTP.
            
            elif endpoint.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                if os.path.exists(self.input_endpoint):
                    return "video"  # Arquivo de vídeo local.
                else:
                    logging.error(f"Arquivo de vídeo não encontrado: {self.input_endpoint}")
                    return None

        logging.warning(f"Tipo de fonte de vídeo não reconhecido: {self.input_endpoint}")
        return None

    def _inject_credentials_rtsp(self, url: str) -> str:
        """Insere credenciais de usuário e senha em uma URL RTSP de forma segura."""
        if not (self.username and self.password and url.lower().startswith("rtsp://")):
            return url

        try:
            # Codifica caracteres especiais no usuário e senha para evitar erros na URL.
            user_enc = quote(self.username, safe="")
            pwd_enc = quote(self.password, safe="")

            # Decompõe a URL para modificar a parte de localização de rede (`netloc`).
            parsed_url = urlparse(url)
            
            # Remonta o `netloc` com as credenciais: "usuario:senha@host:porta".
            netloc_with_creds = f"{user_enc}:{pwd_enc}@{parsed_url.hostname}"
            if parsed_url.port:
                netloc_with_creds += f":{parsed_url.port}"
            
            # Remonta a URL completa com as credenciais injetadas.
            new_url_parts = parsed_url._replace(netloc=netloc_with_creds)
            return urlunparse(new_url_parts)
            
        except Exception as e:
            logging.error(f"Erro ao montar a URL RTSP com credenciais: {e}")
            return url  # Retorna a URL original em caso de erro.

    def _init_capture(self):
        """Inicializa o objeto `cv2.VideoCapture` para a fonte."""
        endpoint_to_use = self.input_endpoint
        if self.source_type == "rtsp":
            endpoint_to_use = self._inject_credentials_rtsp(self.input_endpoint)
            # Loga a URL sem a senha por segurança.
            logging.info(f"Conectando ao stream: {endpoint_to_use.replace(self.password, '*****')}")

        # Converte para int se for uma câmera, senão usa a string do endpoint.
        capture_arg = int(endpoint_to_use) if self.source_type == "camera" else endpoint_to_use
        self.cap = cv2.VideoCapture(capture_arg)
        
        if not self.cap.isOpened():
            logging.error(f"Não foi possível abrir a fonte de vídeo: {self.input_endpoint}")
            self.cap = None  # Define como None para acionar a lógica de reconexão.

    def get_frame(self) -> np.ndarray | None:
        """Obtém um único frame da fonte de vídeo configurada."""
        # --- Fontes Contínuas (OpenCV) ---
        if self.source_type in ("video", "camera", "rtsp"):
            # Se a captura não estiver aberta, tenta reabrir.
            if self.cap is None or not self.cap.isOpened():
                logging.warning(f"Fonte '{self.input_endpoint}' indisponível. Tentando reconectar...")
                self.release()  # Garante a liberação de recursos antigos.
                self._init_capture()
                if self.cap is None:  # Se a reconexão falhar, retorna None.
                    time.sleep(2) # Pausa antes de tentar novamente no próximo ciclo
                    return None

            ret, frame = self.cap.read()
            return frame if ret else None

        # --- Fonte de Snapshot (HTTP) ---
        elif self.source_type == "http":
            try:
                auth = HTTPBasicAuth(self.username, self.password) if self.username else None
                response = requests.get(self.input_endpoint, auth=auth, timeout=self.timeout)
                response.raise_for_status()  # Lança uma exceção para códigos de erro HTTP (4xx ou 5xx).
                
                # Decodifica o conteúdo da resposta (imagem) para um array numpy (frame OpenCV).
                return cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)

            except requests.RequestException as e:
                logging.error(f"Erro de conexão ao buscar snapshot HTTP de '{self.input_endpoint}': {e}")

        return None

    def release(self):
        """Libera o recurso de captura de vídeo."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logging.info(f"Recurso de captura para '{self.input_endpoint}' liberado.")
    
    def __del__(self):
        """Destrutor que garante a liberação dos recursos quando o objeto é destruído."""
        self.release()