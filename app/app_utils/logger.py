import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import os

# Um formato de log mais completo que inclui o nome do processo,
# o que é extremamente útil para depurar aplicações multiprocesso.
LOG_FORMAT = "%(asctime)s — %(processName)s — %(levelname)s — %(message)s"

def setup_logger(logs_save_dir: Path):
    """
    Configura o logger raiz para ser seguro para múltiplos processos.

    Esta função pode ser chamada com segurança no início de cada processo
    para garantir que todos configurem o logging da mesma forma.
    """
    # Pega o logger raiz. Todos os loggers da aplicação herdarão dele.
    os.makedirs(logs_save_dir, exist_ok=True)
    logger = logging.getLogger()

    # Define o nível mínimo de log que será processado.
    # Use logging.DEBUG para ver mensagens mais detalhadas.
    logger.setLevel(logging.INFO)

    # Limpa quaisquer handlers que possam ter sido herdados do processo pai.
    # Isso é crucial para evitar duplicação de logs ou conflitos.
    if logger.hasHandlers():
        logger.handlers.clear()

    # 1. Handler para o console (terminal)
    #    Para que você possa ver os logs em tempo real.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)

    # 2. Handler para arquivos, com rotação por tempo
    #    Salva os logs em um arquivo que é rotacionado à meia-noite.
    #    'backupCount=5' guarda os logs dos últimos 5 dias.
    log_file_path = logs_save_dir / "plate_recognition.log"
    file_handler = TimedRotatingFileHandler(
        log_file_path, 
        when='midnight', 
        interval=1, 
        backupCount=5, 
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)

    logging.info("Logger configurado com sucesso.")