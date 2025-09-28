import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv
import torch

# from app_utils.logger import setup_logger

# --- Constantes de Caminho ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
APP_DIR = ROOT_DIR / 'app'
load_dotenv(ROOT_DIR / ".env")

# --- Funções Auxiliares de Configuração (sem alterações) ---
def get_env_path(base_dir: Path, env_var: str, default: str = None) -> Path | None:
    path_str = os.getenv(env_var, default)
    return base_dir / path_str if path_str else None

def get_env_bool(env_var: str, default: bool = False) -> bool:
    return os.getenv(env_var, str(default)).lower() in ('true', '1', 't', 'y', 'yes')

def get_env_int(env_var: str, default: int = 0) -> int:
    return int(os.getenv(env_var, str(default)))

# --- Dataclass para Agrupar Configurações ---
@dataclass(frozen=True)
class AppSettings:
    """Agrupa todas as configurações da aplicação de forma imutável."""
    # ... (outros campos permanecem os mesmos) ...
    CONFIG_FILE: Path = field(default_factory=lambda: ROOT_DIR / os.getenv("CONFIG_FILE", "config.json"))
    LOGS_SAVE_DIR: Path = field(default_factory=lambda: ROOT_DIR / os.getenv("LOGS_SAVE_DIR", "log/"))
    LOGS_SAVE_DAYS: int = field(default_factory=lambda: get_env_int("LOGS_SAVE_DAYS", 30))
    DB_CONNECTION: Path = field(default_factory=lambda: ROOT_DIR / os.getenv("DB_CONNECTION", "db/captures.db"))
    CAPTURES_SAVE_DIR: Path = field(default_factory=lambda: get_env_path(ROOT_DIR, "CAPTURES_SAVE_DIR"))
    SAVE_SUSPECT_DETECTIONS: bool = field(default_factory=lambda: get_env_bool("SAVE_SUSPECT_DETECTIONS"))
    SUSPECT_DETECTIONS_SAVE_DIR: Path = field(default_factory=lambda: get_env_path(ROOT_DIR, "SUSPECT_DETECTIONS_SAVE_DIR"))
    PLATE_DETECTION_MODEL: Path = field(default_factory=lambda: get_env_path(ROOT_DIR, "PLATE_DETECTION_MODEL"))
    OCR_CHAR_DICT_FILE: Path = field(default_factory=lambda: get_env_path(ROOT_DIR, "OCR_CHAR_DICT_FILE"))
    CHAR_CORRECTIONS_FILE: Path = field(default_factory=lambda: get_env_path(ROOT_DIR, "CHAR_CORRECTIONS_FILE"))
    USE_OCR_DETECTION: bool = field(default_factory=lambda: get_env_bool("USE_OCR_DETECTION"))
    USE_OCR_HPI: bool = field(default_factory=lambda: get_env_bool("USE_OCR_HPI", True))
    USE_OCR_TENSORRT: bool = field(default_factory=lambda: get_env_bool("USE_OCR_TENSORRT", True))
    USE_CONTINUOUS_TRIES: bool = field(default_factory=lambda: get_env_bool("USE_CONTINUOUS_TRIES", False))
    SHOW_CAPTURES: bool = field(default_factory=lambda: get_env_bool("SHOW_CAPTURES"))
    CALCULATE_FPS: bool = field(default_factory=lambda: get_env_bool("CALCULATE_FPS", False))
    DEBUG: bool = field(default_factory=lambda: get_env_bool("DEBUG", False))
    CROP_MARGIN: int = field(default_factory=lambda: get_env_int("CROP_MARGIN"))
    READING_FORMATS: list[str] = field(default_factory=lambda: os.getenv("READING_FORMATS", "").split(","))
    READINGS_FILTER_REGEX: str | None = field(default_factory=lambda: os.getenv("READINGS_FILTER_REGEX"))
    MAX_NO_FRAME_COUNT: int = field(default_factory=lambda: get_env_int("MAX_NO_FRAME_COUNT", 10))
    API_ENDPOINT: str | None = field(default_factory=lambda: os.getenv("API_ENDPOINT"))
    API_USER: str | None = field(default_factory=lambda: os.getenv("API_USER"))
    API_PASSWORD: str | None = field(default_factory=lambda: os.getenv("API_PASSWORD"))
    CLOSE_API_ENDPOINT: str | None = field(default_factory=lambda: os.getenv("CLOSE_API_ENDPOINT"))

    PLATE_DETECTION_DEVICE: str | None = field(default_factory=lambda: os.getenv("PLATE_DETECTION_DEVICE"))
    OCR_DEVICE: str | None = field(default_factory=lambda: os.getenv("OCR_DEVICE"))

    STABILITY_MAX_COORDINATE_DIFFERENCE: int = field(default_factory=lambda: get_env_int("STABILITY_MAX_COORDINATE_DIFFERENCE"))
    STATIONARY_FRAME_THRESHOLD: int = field(default_factory=lambda: get_env_int("STATIONARY_FRAME_THRESHOLD"))

    # Carregados de arquivos ou determinados dinamicamente
    INPUT_SOURCES: dict = field(init=False)
    CHAR_CORRECTIONS: dict = field(init=False)
    
    # Modelos de OCR (lógica de seleção)
    OCR_DETECTION_MODEL: Path | None = field(init=False)
    OCR_RECOGNITION_MODEL: Path | None = field(init=False)
    OCR_CLASSIFICATION_MODEL: Path | None = field(init=False)

    def __post_init__(self):
        """Carrega e valida configurações que dependem de outras ou de arquivos."""

        # setup_logger()

        # Carrega dados do config.json
        with open(self.CONFIG_FILE) as f:
            config_data = json.load(f)

        # 2. Processa e valida cada fonte de entrada
        validated_sources = {}
        sources_from_config = config_data.get("input_sources", {})
        # if not sources_from_config:
            # logging.error("Nenhuma 'input_sources' encontrada no config.json!")
        
        for name, data in sources_from_config.items():
            
            # Atualiza o dicionário da fonte com o dispositivo validado
            updated_data = data.copy()
            validated_sources[name] = updated_data
        
        object.__setattr__(self, 'INPUT_SOURCES', validated_sources)

        # Carrega correções de caracteres
        with open(self.CHAR_CORRECTIONS_FILE, 'r', encoding='utf-8') as f:
            object.__setattr__(self, 'CHAR_CORRECTIONS', json.load(f))

        det_model = get_env_path(ROOT_DIR, "PADDLE_OCR_DET_MODEL")
        rec_model = get_env_path(ROOT_DIR, "PADDLE_OCR_REC_MODEL")

        object.__setattr__(self, 'OCR_RECOGNITION_MODEL', rec_model)
        object.__setattr__(self, 'OCR_DETECTION_MODEL', det_model if self.USE_OCR_DETECTION else None)


# --- Instância Única de Configurações ---
settings = AppSettings()
