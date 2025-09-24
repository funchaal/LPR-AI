import logging
from multiprocessing import Process, set_start_method
import os

# Módulos da aplicação
from app_utils.logger import setup_logger
from app_utils.config import settings
from modules.engine import process_source

# Módulos da aplicação
from modules.detector import load_yolo
from modules.ocr import init_ocr

from app_utils.validate_and_normalize_device import validate_and_normalize_device

def main():
    """
    Função gerente que cria e gerencia os processos para cada fonte de vídeo.
    """
    # Configura o logger principal (já feito no config.py, mas não custa garantir)
    setup_logger()
    
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass # O método de início já pode ter sido definido

    input_sources = settings.INPUT_SOURCES
    num_sources = len(input_sources)
    
    if not settings.INPUT_SOURCES:
        logging.error("Nenhuma fonte de entrada (input_sources) foi definida e validada. Verifique o config.json.")
        return
    
    logging.info(f"Iniciando aplicação com {len(settings.INPUT_SOURCES)} fonte(s) de vídeo.")
    
    processes = []

    yolo_device = settings.PLATE_DETECTION_DEVICE
    ocr_device = settings.OCR_DEVICE

    yolo_validated_device = validate_and_normalize_device(yolo_device)
    yolo_inference_device = '0' if yolo_validated_device == 'gpu' else yolo_validated_device.replace('gpu:', '')

    yolo = load_yolo(settings.PLATE_DETECTION_MODEL, yolo_validated_device)

    ocr_validated_device = validate_and_normalize_device(ocr_device)

    ocr_common_args = {
        'det_model_dir': str(settings.OCR_DETECTION_MODEL) if settings.OCR_DETECTION_MODEL else None,
        'rec_model_dir': str(settings.OCR_RECOGNITION_MODEL) if settings.OCR_RECOGNITION_MODEL else None,
        'use_det': settings.USE_OCR_DETECTION,
        'device': ocr_validated_device, 
        'char_dict_file': settings.OCR_CHAR_DICT_FILE
    }

    ocr = init_ocr(**ocr_common_args)

    if num_sources == 1:
        # --- MODO DE PROCESSO ÚNICO ---
        # Se houver apenas uma fonte, executa diretamente no processo principal.
        
        # Pega o nome e os dados da única fonte de entrada
        input_name, data = next(iter(input_sources.items()))
        
        instance_id = data["instance"]
        input_endpoint = data["input-endpoint"]
        input_user = data.get("input-user")
        input_password = data.get("input-password")
        polygons = data.get("polygons")
        
        logging.info(f"Iniciando processamento para a única fonte: '{input_name}' (Fonte: {input_endpoint})")

        # Chama a função alvo diretamente
        process_source(
            instance_id, input_name, input_endpoint, input_user, input_password, polygons, yolo, ocr, yolo_inference_device
        )

    elif num_sources > 1:
        # --- MODO DE MULTIPROCESSAMENTO ---
        # Se houver mais de uma fonte, cria um processo para cada uma.
        processes = []
        logging.info(f"Múltiplas fontes detectadas ({num_sources}). Iniciando em modo de multiprocessamento.")

        for input_name, data in input_sources.items():
            instance_id = data["instance"]
            input_endpoint = data["input-endpoint"]
            input_user = data.get("input-user")
            input_password = data.get("input-password")
            polygons = data.get("polygons")

            logging.info(f"Criando processo para: '{input_name}' (Fonte: {input_endpoint})")

            process = Process(
                target=process_source,
                args=(instance_id, input_name, input_endpoint, input_user, input_password, polygons, ocr, yolo, yolo_inference_device)
            )
            processes.append(process)
            process.start()
        
        # Aguarda todos os processos terminarem
        for p in processes:
            p.join()

    else:
        # --- NENHUMA FONTE ---
        logging.info("Nenhuma fonte de entrada configurada. Aplicação será encerrada.")

    logging.info("Todos os processamentos foram finalizados. Encerrando a aplicação.")

if __name__ == '__main__':
    main()
