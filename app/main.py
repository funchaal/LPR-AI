# main.py
import logging
from multiprocessing import Process, set_start_method

import os

from pathlib import Path

# Módulos da aplicação
from app_utils.logger import setup_logger
from app_utils.config import settings  # Importa o objeto de configurações já pronto
from modules.engine import process_source # Importa a função de processamento

from app_utils.model_optimizer import ensure_best_model

def main():
    """
    Função gerente que cria e gerencia os processos para cada fonte de vídeo.
    """
    # Configura o logger principal
    setup_logger(settings.LOGS_SAVE_DIR)
    
    logging.info("Verificando e otimizando o modelo YOLO antes de iniciar os processos...")
    optimized_model_path = ensure_best_model() # Não precisa mais de argumentos!

    os.environ['_FINAL_COMPUTE_DEVICE'] = settings.COMPUTE_DEVICE
    
    # Garante compatibilidade entre multiprocessing e CUDA
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    if not settings.INPUT_SOURCES:
        logging.error("Nenhuma fonte de entrada (input_sources) foi definida no arquivo de configuração.")
        return

    logging.info(f"Iniciando aplicação com {len(settings.INPUT_SOURCES)} fonte(s) de vídeo.")
    
    processes = []
    for input_name, data in settings.INPUT_SOURCES.items():
        instance_id = data["instance"]
        input_endpoint = data["input_endpoint"]
        polygons = data.get("polygons")
        
        logging.info(f"Criando processo para: {input_name} (Fonte: {input_endpoint})")
        
        process = Process(
            target=process_source, 
            args=(instance_id, input_name, input_endpoint, optimized_model_path, polygons)
        )
        processes.append(process)
        process.start()
        
    # Aguarda todos os processos terminarem
    for p in processes:
        p.join()
        
    logging.info("Todos os processos foram finalizados. Encerrando a aplicação.")

if __name__ == '__main__':
    main()