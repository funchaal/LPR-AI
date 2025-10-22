import logging
import sys
from logging.handlers import TimedRotatingFileHandler
import os
from app_utils.config import settings

# Um formato de log mais completo que inclui o nome do processo,
# o que é extremamente útil para depurar aplicações multiprocesso.
LOG_FORMAT = "%(asctime)s — [%(instance_id)s][%(input_name)s] — %(levelname)s — %(message)s"

class ContextFilter(logging.Filter):
    def __init__(self, instance_id, input_name):
        super().__init__()
        self.instance_id = instance_id
        self.input_name = input_name

    def filter(self, record):
        record.instance_id = self.instance_id
        record.input_name = self.input_name
        return True

def setup_logger(input_name="MAIN"):
    """
    Configura um logger específico para uma instância, com um arquivo de log dedicado.

    Args:
        instance_id (str): O identificador da instância.
        input_name (str): O nome da entrada associada à instância.
    """
    instance_id = settings.INSTANCE_ID
    logs_save_dir = settings.LOGS_SAVE_DIR
    os.makedirs(logs_save_dir, exist_ok=True)

    # Cria um nome de logger único para cada instância para evitar conflitos
    logger_name = f"instance_{instance_id}"
    logger = logging.getLogger(logger_name)
    logger.propagate = False

    if settings.DEBUG:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    # Adiciona o filtro com os dados da instância
    context_filter = ContextFilter(instance_id, input_name)
    logger.addFilter(context_filter)

    # Handler para o console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)

    # Handler para arquivo, com nome de arquivo dinâmico
    log_file_path = logs_save_dir / f"{instance_id}.log"
    file_handler = TimedRotatingFileHandler(
        log_file_path,
        when='midnight',
        interval=1,
        backupCount=settings.LOGS_SAVE_DAYS,
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)

    logger.info("Logger configurado com sucesso para a instância.")
    return logger