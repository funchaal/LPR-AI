import cv2
import requests
import numpy as np
import logging
import os
import time
import warnings
from requests.auth import HTTPBasicAuth
from urllib.parse import urlparse, urlunparse, quote

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# ==========================
# Configuração do logging
# ==========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(processName)s — %(levelname)s — %(message)s"
)

# Desativa avisos de "corrupt JPEG data"
warnings.filterwarnings("ignore", category=UserWarning, module="cv2")


class VideoSource:
    """
    Classe para capturar frames de diferentes fontes:
    - Câmera (via índice da webcam, ex: 0, 1, 2...)
    - Vídeo local (mp4, avi, mkv, etc.)
    - Stream HTTP (snapshot único em JPG)
    - Stream RTSP/MJPEG contínuo
    """

    def __init__(self, input_endpoint, username=None, password=None, timeout=3):
        self.input_endpoint = input_endpoint
        self.username = username
        self.password = password
        self.timeout = timeout
        self.cap = None
        self.source_type = self._detect_source_type()

        if self.source_type in ["video", "camera", "rtsp"]:
            self._init_capture()

    def _detect_source_type(self):
        """Detecta automaticamente o tipo da fonte de vídeo."""
        if isinstance(self.input_endpoint, int) or str(self.input_endpoint).isdigit():
            return "camera"

        if isinstance(self.input_endpoint, str):
            endpoint = self.input_endpoint.lower()
            # Trata RTSP e MJPEG como fluxos contínuos via OpenCV
            if endpoint.startswith("rtsp://") or endpoint.endswith(".mjpg"):
                return "rtsp"
            elif endpoint.startswith(("http://", "https://")):
                return "http"  # snapshots em JPG via requests
            elif os.path.exists(endpoint) and endpoint.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                return "video"

        logging.warning(f"Tipo de input não reconhecido: {self.input_endpoint}")
        return None

    def _inject_credentials_rtsp(self, url):
        """
        Insere usuário e senha na URL de forma segura, tratando caracteres especiais.
        """
        if not (self.username and self.password and url.lower().startswith("rtsp://")):
            return url

        try:
            # Codifica usuário e senha para serem seguros na URL
            user_enc = quote(self.username, safe="")
            pwd_enc = quote(self.password, safe="")

            # Decompõe a URL em suas partes
            parsed_url = urlparse(url)

            # Recria o `netloc` (network location) com as credenciais
            # Ex: "10.151.60.70:554" -> "admin:Dpw%402025@10.151.60.70:554"
            netloc_with_creds = f"{user_enc}:{pwd_enc}@{parsed_url.hostname}"
            if parsed_url.port:
                netloc_with_creds += f":{parsed_url.port}"
            
            # Remonta a URL com o novo netloc
            new_url_parts = parsed_url._replace(netloc=netloc_with_creds)
            return urlunparse(new_url_parts)
            
        except Exception as e:
            logging.error(f"Erro ao montar a URL RTSP com credenciais: {e}")
            return url


    def _init_capture(self):
        """Inicializa o OpenCV para vídeo/câmera/RTSP."""
        endpoint_to_use = self.input_endpoint
        if self.source_type == "rtsp":
            endpoint_to_use = self._inject_credentials_rtsp(self.input_endpoint)
            logging.info(f"Tentando conectar ao stream: {endpoint_to_use.replace(self.password, '*****')}")

        self.cap = cv2.VideoCapture(
            int(endpoint_to_use) if self.source_type == "camera" else endpoint_to_use
        )
        if not self.cap.isOpened():
            logging.error(f"Erro ao abrir a fonte de vídeo: {self.input_endpoint}")
            # Não lança exceção para permitir tentativas de reconexão no loop principal
            self.cap = None

    def get_frame(self):
        """Obtém um frame da fonte configurada."""
        if self.source_type in ("video", "camera", "rtsp"):
            if self.cap is None or not self.cap.isOpened():
                logging.warning("Fonte de vídeo indisponível, tentando reabrir...")
                self.release()  # Garante que o recurso antigo foi liberado
                self._init_capture()
                # Se ainda assim falhar, retorna None
                if self.cap is None:
                    return None

            ret, frame = self.cap.read()
            return frame if ret else None

        elif self.source_type == "http":
            try:
                auth = HTTPBasicAuth(self.username, self.password) if self.username else None
                response = requests.get(self.input_endpoint, auth=auth, timeout=self.timeout)
                response.raise_for_status() # Lança erro para códigos HTTP 4xx/5xx
                
                return cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)

            except requests.RequestException as e:
                logging.error(f"Erro de conexão HTTP: {e}")

        return None

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def __del__(self):
        # Garante que a câmera seja liberada quando o objeto for destruído
        self.release()
