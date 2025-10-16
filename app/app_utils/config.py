# -*- coding: utf-8 -*-
"""
Módulo de configuração para a aplicação.

Este script é responsável por carregar, validar e disponibilizar todas as
configurações necessárias para a execução da aplicação. As configurações são
carregadas de variáveis de ambiente (de um arquivo .env) e de um arquivo
`config.json`. Utiliza um dataclass imutável (`AppSettings`) para garantir
que as configurações não sejam alteradas durante a execução.
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# --- CONSTANTES DE CAMINHO ---
# Define os diretórios base para a aplicação, garantindo que os caminhos
# sejam sempre relativos à raiz do projeto.
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
APP_DIR = ROOT_DIR / 'app'

# Carrega as variáveis de ambiente do arquivo .env localizado na raiz do projeto.
load_dotenv(ROOT_DIR / ".env")

# --- FUNÇÕES AUXILIARES DE CONFIGURAÇÃO ---

def get_env_path(base_dir: Path, env_var: str, default: str = None) -> Path | None:
    """Obtém um caminho de uma variável de ambiente e o resolve em relação a um diretório base."""
    path_str = os.getenv(env_var, default)
    # Retorna um objeto Path completo se a variável de ambiente for definida.
    return base_dir / path_str if path_str else None

def get_env_bool(env_var: str, default: bool = False) -> bool:
    """Obtém um valor booleano de uma variável de ambiente."""
    # Considera 'true', '1', 't', 'y', 'yes' como True.
    return os.getenv(env_var, str(default)).lower() in ('true', '1', 't', 'y', 'yes')

def get_env_int(env_var: str, default: int = 0) -> int:
    """Obtém um valor inteiro de uma variável de ambiente."""
    return int(os.getenv(env_var, str(default)))

# --- DATACLASS PARA AGRUPAR CONFIGURAÇÕES ---

@dataclass(frozen=True)
class AppSettings:
    """
    Agrupa todas as configurações da aplicação de forma imutável (frozen=True).
    A imutabilidade previne alterações acidentais nas configurações após a inicialização.
    As configurações são carregadas principalmente de variáveis de ambiente.
    """
    # --- Configurações Gerais e de Arquivos ---
    CONFIG_FILE: Path = field(default_factory=lambda: ROOT_DIR / os.getenv("CONFIG_FILE", "config.json"))
    LOGS_SAVE_DIR: Path = field(default_factory=lambda: ROOT_DIR / os.getenv("LOGS_SAVE_DIR", "log/"))
    LOGS_SAVE_DAYS: int = field(default_factory=lambda: get_env_int("LOGS_SAVE_DAYS", 30))
    DB_CONNECTION: Path = field(default_factory=lambda: ROOT_DIR / os.getenv("DB_CONNECTION", "db/captures.db"))
    CAPTURES_SAVE_DIR: Path = field(default_factory=lambda: get_env_path(ROOT_DIR, "CAPTURES_SAVE_DIR"))
    SAVE_SUSPECT_DETECTIONS: bool = field(default_factory=lambda: get_env_bool("SAVE_SUSPECT_DETECTIONS"))
    SUSPECT_DETECTIONS_SAVE_DIR: Path = field(default_factory=lambda: get_env_path(ROOT_DIR, "SUSPECT_DETECTIONS_SAVE_DIR"))
    CHAR_CORRECTIONS_FILE: Path = field(default_factory=lambda: get_env_path(ROOT_DIR, "CHAR_CORRECTIONS_FILE"))

    # --- Configurações dos Modelos de IA ---
    PLATE_DETECTION_MODEL: Path = field(default_factory=lambda: get_env_path(ROOT_DIR, "PLATE_DETECTION_MODEL"))
    PLATE_DETECTION_DEVICE: str | None = field(default_factory=lambda: os.getenv("PLATE_DETECTION_DEVICE"))
    OCR_DEVICE: str | None = field(default_factory=lambda: os.getenv("OCR_DEVICE"))
    OCR_CHAR_DICT_FILE: Path = field(default_factory=lambda: get_env_path(ROOT_DIR, "OCR_CHAR_DICT_FILE"))
    USE_OCR_DETECTION: bool = field(default_factory=lambda: get_env_bool("USE_OCR_DETECTION"))

    # --- Configurações de Processamento e Lógica ---
    USE_CONTINUOUS_TRIES: bool = field(default_factory=lambda: get_env_bool("USE_CONTINUOUS_TRIES", False))
    SHOW_CAPTURES: bool = field(default_factory=lambda: get_env_bool("SHOW_CAPTURES"))
    CALCULATE_FPS: bool = field(default_factory=lambda: get_env_bool("CALCULATE_FPS", False))
    FPS_LOG_LIMIAR: int = field(default_factory=lambda: get_env_int("FPS_LOG_LIMIAR", 5))
    DEBUG: bool = field(default_factory=lambda: get_env_bool("DEBUG", False))
    CROP_MARGIN: int = field(default_factory=lambda: get_env_int("CROP_MARGIN"))
    READING_FORMATS: list[str] = field(default_factory=lambda: os.getenv("READING_FORMATS", "").split(","))
    READINGS_FILTER_REGEX: str | None = field(default_factory=lambda: os.getenv("READINGS_FILTER_REGEX"))
    MAX_NO_FRAME_COUNT: int = field(default_factory=lambda: get_env_int("MAX_NO_FRAME_COUNT", 10))
    STABILITY_MAX_COORDINATE_DIFFERENCE: int = field(default_factory=lambda: get_env_int("STABILITY_MAX_COORDINATE_DIFFERENCE"))
    STATIONARY_FRAME_THRESHOLD: int = field(default_factory=lambda: get_env_int("STATIONARY_FRAME_THRESHOLD"))

    # --- Configurações de API Externa ---
    API_ENDPOINT: str | None = field(default_factory=lambda: os.getenv("API_ENDPOINT"))
    API_USER: str | None = field(default_factory=lambda: os.getenv("API_USER"))
    API_PASSWORD: str | None = field(default_factory=lambda: os.getenv("API_PASSWORD"))
    CLOSE_API_ENDPOINT: str | None = field(default_factory=lambda: os.getenv("CLOSE_API_ENDPOINT"))

    # --- Campos Carregados Dinamicamente (init=False) ---
    # Estes campos são inicializados no método __post_init__.
    INPUT_SOURCES: dict = field(init=False)
    CHAR_CORRECTIONS: dict = field(init=False)
    OCR_DETECTION_MODEL: Path | None = field(init=False)
    OCR_RECOGNITION_MODEL: Path | None = field(init=False)

    def __post_init__(self):
        """Executa após a inicialização do dataclass para carregar configurações complexas."""
        # O uso de `object.__setattr__` é necessário porque o dataclass é imutável (frozen=True).
        # Este método permite atribuir valores aos campos `init=False` após a criação da instância.

        # Carrega as fontes de entrada do arquivo `config.json`.
        try:
            with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except FileNotFoundError:
            logging.error(f"Arquivo de configuração '{self.CONFIG_FILE}' não encontrado.")
            config_data = {}
        except json.JSONDecodeError:
            logging.error(f"Erro ao decodificar o JSON do arquivo '{self.CONFIG_FILE}'.")
            config_data = {}

        # Processa e valida cada fonte de entrada do `config.json`.
        validated_sources = {}
        sources_from_config = config_data.get("input_sources", {})
        if not sources_from_config:
            logging.warning("Nenhuma 'input_sources' encontrada no config.json!")
        
        for name, data in sources_from_config.items():
            # TODO: Adicionar lógica de validação para cada fonte (ex: verificar se o endpoint é acessível).
            updated_data = data.copy()
            validated_sources[name] = updated_data
        
        object.__setattr__(self, 'INPUT_SOURCES', validated_sources)

        # Carrega o arquivo de correções de caracteres (ex: trocar 'O' por '0').
        try:
            with open(self.CHAR_CORRECTIONS_FILE, 'r', encoding='utf-8') as f:
                object.__setattr__(self, 'CHAR_CORRECTIONS', json.load(f))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.warning(f"Não foi possível carregar o arquivo de correções de caracteres: {e}")
            object.__setattr__(self, 'CHAR_CORRECTIONS', {})

        # Define os caminhos para os modelos de OCR (detecção e reconhecimento).
        det_model = get_env_path(ROOT_DIR, "PADDLE_OCR_DET_MODEL")
        rec_model = get_env_path(ROOT_DIR, "PADDLE_OCR_REC_MODEL")

        object.__setattr__(self, 'OCR_RECOGNITION_MODEL', rec_model)
        # O modelo de detecção de texto do OCR só é carregado se a opção `USE_OCR_DETECTION` for verdadeira.
        object.__setattr__(self, 'OCR_DETECTION_MODEL', det_model if self.USE_OCR_DETECTION else None)

# --- INSTÂNCIA ÚNICA DE CONFIGURAÇÕES ---
# Cria uma instância única e imutável das configurações que será importada
# por outros módulos da aplicação.
settings = AppSettings()