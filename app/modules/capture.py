import cv2
import requests
import numpy as np
import logging
import os

def detect_source_type(input_endpoint):
    if isinstance(input_endpoint, int) or str(input_endpoint).isdigit():
        return "camera"
    elif isinstance(input_endpoint, str):
        if input_endpoint.startswith(('http://', 'https://', 'rtsp://')):
            return "stream"
        elif os.path.exists(input_endpoint) and input_endpoint.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            return "video"
    logging.warning(f"Tipo de input não reconhecido: {input_endpoint}")
    return None

def init_capture(input_endpoint):
    source_type = detect_source_type(input_endpoint)

    if source_type in ["video", "camera"]:
        cap = cv2.VideoCapture(input_endpoint)
        if not cap.isOpened():
            logging.error(f"Erro ao abrir a fonte de vídeo: {input_endpoint}")
            raise RuntimeError("Falha ao abrir vídeo/câmera")
        return cap, source_type

    # stream (ex: imagem periódica via HTTP)
    return None, source_type

def get_frame(source_type, cap, input_endpoint):
    if source_type in ["video", "camera"]:
        ret, frame = cap.read()
        return frame if ret else None
    elif source_type == "stream":
        try:
            content = requests.get(input_endpoint, timeout=2)
            if content.status_code == 200:
                return cv2.imdecode(np.frombuffer(content.content, np.uint8), cv2.IMREAD_COLOR)
            else:
                logging.error(f"Erro ao obter frame da stream: {input_endpoint} - HTTP {content.status_code}")
        except requests.RequestException as e:
            logging.error(f"Erro de conexão com stream: {e}")
    return None
