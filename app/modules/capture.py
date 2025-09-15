import cv2
import requests
import numpy as np
import logging
import os
from requests.auth import HTTPBasicAuth

class VideoSource:
    """
    Classe para capturar frames de diferentes fontes:
    - Câmera (via índice da webcam, ex: 0, 1, 2...)
    - Vídeo local (mp4, avi, mkv, etc.)
    - Stream HTTP/RTSP (com suporte a autenticação)
    """

    def __init__(self, input_endpoint, username=None, password=None, timeout=3):
        """
        :param input_endpoint: caminho do vídeo, índice da câmera ou URL da stream
        :param username: login para autenticação (caso necessário)
        :param password: senha para autenticação (caso necessário)
        :param timeout: tempo limite para requests HTTP
        """
        self.input_endpoint = input_endpoint
        self.username = username
        self.password = password
        self.timeout = timeout
        self.cap = None
        self.source_type = self._detect_source_type()

        if self.source_type in ["video", "camera"]:
            self._init_capture()

    def _detect_source_type(self):
        """Detecta automaticamente o tipo da fonte de vídeo."""
        if isinstance(self.input_endpoint, int) or str(self.input_endpoint).isdigit():
            return "camera"
        elif isinstance(self.input_endpoint, str):
            endpoint = self.input_endpoint.lower()
            if endpoint.startswith("rtsp://") or endpoint.endswith(".mjpg"):
                return "rtsp"   # fluxo contínuo via cv2.VideoCapture
            elif endpoint.startswith(("http://", "https://")):
                return "http"   # snapshots em JPG
            elif os.path.exists(endpoint) and endpoint.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                return "video"

        logging.warning(f"Tipo de input não reconhecido: {self.input_endpoint}")
        return None

    def _init_capture(self):
        """Inicializa o OpenCV para vídeo/câmera/RTSP."""
        if self.source_type in ("video", "camera", "rtsp"):
            self.cap = cv2.VideoCapture(
                int(self.input_endpoint) if self.source_type == "camera" else self.input_endpoint
            )
            if not self.cap.isOpened():
                logging.error(f"Erro ao abrir a fonte de vídeo: {self.input_endpoint}")
                raise RuntimeError("Falha ao abrir vídeo/câmera/stream RTSP")

    def get_frame(self):
        """Obtém um frame da fonte configurada."""
        if self.source_type in ("video", "camera", "rtsp"):
            ret, frame = self.cap.read()
            return frame if ret else None

        elif self.source_type == "http":
            try:
                auth = HTTPBasicAuth(self.username, self.password) if self.username and self.password else None
                response = requests.get(self.input_endpoint, auth=auth, timeout=self.timeout)
                if response.status_code == 200:
                    return cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
                else:
                    logging.error(f"Erro HTTP {response.status_code} ao obter frame: {self.input_endpoint}")
            except requests.RequestException as e:
                logging.error(f"Erro de conexão HTTP: {e}")

        return None


    def release(self):
        """Libera recursos da captura."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
