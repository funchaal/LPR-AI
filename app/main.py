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

# Módulos da aplicação
from app_utils.logger import setup_logger
from app_utils.config import settings
from modules.engine import process_source

def main():
    """
    Função principal que orquestra a inicialização e o processamento das fontes de vídeo.

    Esta função realiza as seguintes etapas:
    1. Configura o logger da aplicação.
    2. Define o método de início para multiprocessamento como 'spawn' para compatibilidade entre plataformas.
    3. Carrega e valida as fontes de entrada a partir da configuração.
    4. Inicia o processamento:
        - Se uma única fonte for fornecida, executa em modo de thread única.
        - Se múltiplas fontes forem fornecidas, cria um processo separado para cada uma.
    5. Aguarda a conclusão de todos os processos antes de encerrar.
    """
    # Carrega as fontes de entrada (ex: câmeras RTSP, arquivos de vídeo) do arquivo de configuração.
    logger = setup_logger(input_name="MAIN")

    input_sources = settings.INPUT_SOURCES
    num_sources = len(input_sources)

    # Verifica se alguma fonte de entrada foi configurada. Se não, encerra a aplicação.
    if not input_sources:
        logger.info("Nenhuma fonte de entrada (input_sources) foi definida no config.json. A aplicação será encerrada.")
        return

    logger.info(f"Aplicação iniciada com {num_sources} fonte(s) de vídeo.")

    # --- GERENCIAMENTO DE PROCESSOS ---
    if num_sources == 1:
        # MODO DE PROCESSO ÚNICO: Executa diretamente no processo principal.
        logger.info("Iniciando em modo de processo único.")
        
        # Extrai os dados da única fonte de entrada.
        input_name, data = next(iter(input_sources.items()))
        input_endpoint = data["input-endpoint"]
        input_username = data.get("input-username")
        input_password = data.get("input-password")
        visible_polygons = data.get("visible-polygons")

        logger.info(f"Iniciando processamento para a fonte: '{input_name}' (Endpoint: {input_endpoint})")

        # Chama a função de processamento diretamente.
        process_source(
            logger,
            input_name, input_endpoint, input_username, input_password,
            visible_polygons
        )

    elif num_sources > 1:
        # MODO DE MULTIPROCESSAMENTO: Cria um processo para cada fonte.
        
        from multiprocessing import Process, set_start_method

        # Tenta definir o método de início de multiprocessamento como 'spawn'.
        # 'spawn' cria um novo processo do zero, o que é mais seguro e compatível
        # com diferentes sistemas operacionais, especialmente ao usar GPUs.
        try:
            set_start_method('spawn')
        except RuntimeError:
            # Se o método já foi definido, ignora o erro.
            logger.info("O método de início de multiprocessamento já foi definido.")
            pass

        logger.info(f"Múltiplas fontes detectadas ({num_sources}). Iniciando em modo de multiprocessamento.")
        processes = []

        for input_name, data in input_sources.items():
            input_endpoint = data["input-endpoint"]
            input_username = data.get("input-username")
            input_password = data.get("input-password")
            visible_polygons = data.get("visible-polygons")

            logger.info(f"Criando processo para a fonte: '{input_name}' (Endpoint: {input_endpoint})")

            # Cria um novo processo para cada fonte de vídeo.
            # Isso permite o processamento paralelo, melhorando o desempenho.
            process = Process(
                target=process_source,
                args=(
                    logger,
                    input_name, input_endpoint, input_username, input_password,
                    visible_polygons
                )
            )
            processes.append(process)
            process.start()
        
        # Aguarda a finalização de todos os processos criados.
        logger.info("Aguardando a finalização de todos os processos...")
        for p in processes:
            p.join()

    logger.info("Todos os processamentos foram finalizados. Encerrando a aplicação.")

if __name__ == '__main__':
    # Garante que a função main() seja chamada apenas quando o script é executado diretamente.
    main()