import json
import logging

def load_config(path):
    try:
        with open(path, 'r') as f:
            config = json.load(f)
        logging.info(f"Configuração carregada de {path}")
        return config
    except Exception as e:
        logging.error(f"Erro ao carregar configuração: {e}")
        raise