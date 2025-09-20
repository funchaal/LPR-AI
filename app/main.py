import logging
from multiprocessing import Process, set_start_method
import os

# Módulos da aplicação
from app_utils.logger import setup_logger
from app_utils.config import settings
from modules.engine import process_source

def main():
    """
    Função gerente que cria e gerencia os processos para cada fonte de vídeo.
    """
    # Configura o logger principal (já feito no config.py, mas não custa garantir)
    setup_logger(settings.LOGS_SAVE_DIR)
    
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass # O método de início já pode ter sido definido
    
    if not settings.INPUT_SOURCES:
        logging.error("Nenhuma fonte de entrada (input_sources) foi definida e validada. Verifique o config.json.")
        return
    
    logging.info(f"Iniciando aplicação com {len(settings.INPUT_SOURCES)} fonte(s) de vídeo.")
    
    processes = []

    for input_name, data in settings.INPUT_SOURCES.items():
        instance_id = data["instance"]
        input_endpoint = data["input-endpoint"]
        input_user = data.get("username")
        input_password = data.get("password")
        polygons = data.get("polygons")
        yolo_device = data["yolo-device"]
        ocr_device = data["ocr-device"]

        logging.info(f"Criando processo para: '{input_name}' (Fonte: {input_endpoint})")

        process = Process(
            target=process_source,
            args=(instance_id, input_name, input_endpoint, input_user, input_password, polygons, yolo_device, ocr_device)
        )
        processes.append(process)
        process.start()
    
    # Aguarda todos os processos terminarem
    for p in processes:
        p.join()
    
    logging.info("Todos os processos foram finalizados. Encerrando a aplicação.")

if __name__ == '__main__':
    main()
