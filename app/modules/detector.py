# modules/detector.py

import logging
from pathlib import Path
from ultralytics import YOLO

from app.app_utils.optimize_yolo_model import ensure_best_model

def load_yolo(base_model_path: Path, device: str) -> YOLO:
    """
    Carrega um modelo YOLO a partir de um caminho específico.

    Args:
        model_path (Path): O caminho completo para o arquivo do modelo (.pt ou .engine).

    Returns:
        YOLO: Uma instância do modelo YOLO carregado.
    """

    optimized_model_path = ensure_best_model(base_model_path, device)

    try:
        # O 'task' é inferido automaticamente a partir de modelos .engine
        model = YOLO(optimized_model_path, task='detect')
        logging.info(f"Modelo YOLO carregado com sucesso de '{optimized_model_path}'")
        return model
    except Exception as e:
        logging.error(f"Erro CRÍTICO ao carregar modelo YOLO de '{optimized_model_path}': {e}")
        # Se nem o modelo otimizado nem o fallback puderem ser carregados, o programa não pode continuar.
        raise