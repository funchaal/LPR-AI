from ultralytics import YOLO
import logging

def load_yolo(model_path):
    try:
        model = YOLO(model_path, verbose=False)
        logging.info(f"Modelo YOLO carregado de {model_path}")
        return model
    except Exception as e:
        logging.error(f"Erro ao carregar modelo YOLO: {e}")
        raise