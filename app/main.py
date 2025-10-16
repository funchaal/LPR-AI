# -*- coding: utf-8 -*-

"""
Ponto de entrada principal para a aplicação de reconhecimento de placas.

Este script inicializa a configuração, configura o logging, carrega os modelos de IA
(YOLO para detecção de placas e OCR para reconhecimento de caracteres) e gerencia
o processamento de múltiplas fontes de vídeo. Ele pode operar em modo de processo
único (para uma fonte) ou multiprocesso (para múltiplas fontes), garantindo
eficiência e escalabilidade.
"""

import logging
from multiprocessing import Process, set_start_method

# Módulos da aplicação
from app_utils.logger import setup_logger
from app_utils.config import settings
from modules.engine import process_source
from modules.detector import load_yolo
from modules.ocr import init_ocr
from app_utils.validate_and_normalize_device import validate_and_normalize_device

def main():
    """
    Função principal que orquestra a inicialização e o processamento das fontes de vídeo.

    Esta função realiza as seguintes etapas:
    1. Configura o logger da aplicação.
    2. Define o método de início para multiprocessamento como 'spawn' para compatibilidade entre plataformas.
    3. Carrega e valida as fontes de entrada a partir da configuração.
    4. Valida e normaliza os dispositivos de inferência (CPU, GPU) para YOLO e OCR.
    5. Carrega os modelos de detecção de placas (YOLO) e OCR.
    6. Inicia o processamento:
        - Se uma única fonte for fornecida, executa em modo de thread única.
        - Se múltiplas fontes forem fornecidas, cria um processo separado para cada uma.
    7. Aguarda a conclusão de todos os processos antes de encerrar.
    """
    # Inicializa o logger para registrar informações e erros.
    setup_logger()

    # Tenta definir o método de início de multiprocessamento como 'spawn'.
    # 'spawn' cria um novo processo do zero, o que é mais seguro e compatível
    # com diferentes sistemas operacionais, especialmente ao usar GPUs.
    try:
        set_start_method('spawn')
    except RuntimeError:
        # Se o método já foi definido, ignora o erro.
        logging.debug("O método de início de multiprocessamento já foi definido.")
        pass

    # Carrega as fontes de entrada (ex: câmeras RTSP, arquivos de vídeo) do arquivo de configuração.
    input_sources = settings.INPUT_SOURCES
    num_sources = len(input_sources)

    # Verifica se alguma fonte de entrada foi configurada. Se não, encerra a aplicação.
    if not input_sources:
        logging.error("Nenhuma fonte de entrada (input_sources) foi definida no config.json. A aplicação será encerrada.")
        return

    logging.info(f"Aplicação iniciada com {num_sources} fonte(s) de vídeo.")

    # --- CARREGAMENTO E CONFIGURAÇÃO DOS MODELOS ---
    logging.info("Carregando modelos de IA...")

    # Valida e normaliza o dispositivo para o modelo de detecção de placas (YOLO).
    yolo_device = settings.PLATE_DETECTION_DEVICE
    yolo_validated_device = validate_and_normalize_device(yolo_device)
    # O formato do dispositivo para a inferência YOLO pode variar.
    # Por exemplo, '0' para a primeira GPU ou 'cpu' para a CPU.
    yolo_inference_device = '0' if yolo_validated_device == 'gpu' else yolo_validated_device.replace('gpu:', '')

    # Carrega o modelo YOLO com o dispositivo especificado.
    yolo = load_yolo(settings.PLATE_DETECTION_MODEL, yolo_validated_device)

    # Valida e normaliza o dispositivo para o modelo de OCR.
    ocr_device = settings.OCR_DEVICE
    ocr_validated_device = validate_and_normalize_device(ocr_device)

    # Prepara os argumentos comuns para a inicialização do OCR.
    ocr_common_args = {
        'det_model_dir': str(settings.OCR_DETECTION_MODEL) if settings.OCR_DETECTION_MODEL else None,
        'rec_model_dir': str(settings.OCR_RECOGNITION_MODEL) if settings.OCR_RECOGNITION_MODEL else None,
        'use_det': settings.USE_OCR_DETECTION,
        'device': ocr_validated_device,
        'char_dict_file': settings.OCR_CHAR_DICT_FILE
    }

    # Inicializa o motor de OCR.
    ocr = init_ocr(**ocr_common_args)
    logging.info("Modelos de IA carregados com sucesso.")

    # --- GERENCIAMENTO DE PROCESSOS ---
    if num_sources == 1:
        # MODO DE PROCESSO ÚNICO: Executa diretamente no processo principal.
        logging.info("Iniciando em modo de processo único.")
        
        # Extrai os dados da única fonte de entrada.
        input_name, data = next(iter(input_sources.items()))
        instance_id = data["instance"]
        input_endpoint = data["input-endpoint"]
        polygons = data.get("polygons")

        logging.info(f"Iniciando processamento para a fonte: '{input_name}' (Endpoint: {input_endpoint})")

        # Chama a função de processamento diretamente.
        process_source(
            instance_id, input_name, input_endpoint, data.get("input-user"), data.get("input-password"),
            polygons, yolo, ocr, yolo_inference_device
        )

    elif num_sources > 1:
        # MODO DE MULTIPROCESSAMENTO: Cria um processo para cada fonte.
        logging.info(f"Múltiplas fontes detectadas ({num_sources}). Iniciando em modo de multiprocessamento.")
        processes = []

        for input_name, data in input_sources.items():
            instance_id = data["instance"]
            input_endpoint = data["input-endpoint"]
            polygons = data.get("polygons")

            logging.info(f"Criando processo para a fonte: '{input_name}' (Endpoint: {input_endpoint})")

            # Cria um novo processo para cada fonte de vídeo.
            # Isso permite o processamento paralelo, melhorando o desempenho.
            process = Process(
                target=process_source,
                args=(
                    instance_id, input_name, input_endpoint, data.get("input-user"), data.get("input-password"),
                    polygons, yolo, ocr, yolo_inference_device
                )
            )
            processes.append(process)
            process.start()
        
        # Aguarda a finalização de todos os processos criados.
        logging.info("Aguardando a finalização de todos os processos...")
        for p in processes:
            p.join()

    logging.info("Todos os processamentos foram finalizados. Encerrando a aplicação.")

if __name__ == '__main__':
    # Garante que a função main() seja chamada apenas quando o script é executado diretamente.
    main()