# modules/detector.py

import logging
from pathlib import Path
from ultralytics import YOLO

def load_yolo(model_path: Path) -> YOLO:
    """
    Carrega um modelo YOLO a partir de um caminho específico.

    Args:
        model_path (Path): O caminho completo para o arquivo do modelo (.pt ou .engine).

    Returns:
        YOLO: Uma instância do modelo YOLO carregado.
    """
    try:
        # O 'task' é inferido automaticamente a partir de modelos .engine
        model = YOLO(model_path, task='detect')
        logging.info(f"Modelo YOLO carregado com sucesso de '{model_path}'")
        return model
    except Exception as e:
        logging.error(f"Erro CRÍTICO ao carregar modelo YOLO de '{model_path}': {e}")
        # Se nem o modelo otimizado nem o fallback puderem ser carregados, o programa não pode continuar.
        raise