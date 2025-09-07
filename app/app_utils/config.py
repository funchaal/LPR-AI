# config.py
import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

from app_utils.logger import setup_logger
import torch

# --- Constantes de Caminho ---
# Define os diretórios base para que o projeto seja portátil
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
APP_DIR = ROOT_DIR / 'app'
load_dotenv(ROOT_DIR / ".env")

# --- Funções Auxiliares de Configuração ---

def get_env_path(base_dir: Path, env_var: str, default: str = None) -> Path | None:
    """Carrega um caminho de uma variável de ambiente e o resolve em relação a um diretório base."""
    path_str = os.getenv(env_var, default)
    return base_dir / path_str if path_str else None

def get_env_bool(env_var: str, default: bool = False) -> bool:
    """Carrega um booleano de uma variável de ambiente."""
    return os.getenv(env_var, str(default)).lower() in ('true', '1', 't', 'y', 'yes')

def get_env_int(env_var: str, default: int = 0) -> int:
    """Carrega um inteiro de uma variável de ambiente."""
    return int(os.getenv(env_var, str(default)))

def _determine_compute_device() -> str:
    # Verifica se a decisão já foi tomada pelo processo principal
    final_device_override = os.getenv('_FINAL_COMPUTE_DEVICE')
    if final_device_override:
        # Se sim, simplesmente retorna a decisão sem verificar o hardware
        return final_device_override
    
    setup_logger(ROOT_DIR / os.getenv("LOGS_SAVE_DIR", "log/"))

    # O código abaixo só roda no processo principal
    requested_device = os.getenv("COMPUTE_DEVICE", "cpu").lower()
    logging.info(f"Dispositivo solicitado no .env: {requested_device.upper()}")

    if requested_device != 'cuda':
        logging.info("==> O dispositivo 'CPU' será utilizado conforme solicitado.")
        return "cpu"

    is_gpu_available = torch.cuda.is_available()
    is_paddle_gpu_version = False
    try:
        import paddle
        if paddle.is_compiled_with_cuda():
            is_paddle_gpu_version = True
    except ImportError:
        logging.error("FALHA CRÍTICA AO IMPORTAR PADDLEPADDLE. Verifique a instalação.")
    except Exception as e:
        logging.error(f"Erro inesperado ao verificar o PaddlePaddle: {e}")

    logging.info("Verificando compatibilidade para CUDA...")
    logging.info(f"  - Hardware (GPU NVIDIA detectada): {'SIM' if is_gpu_available else 'NÃO'}")
    logging.info(f"  - Software (PaddlePaddle compilado para GPU): {'SIM' if is_paddle_gpu_version else 'NÃO'}")

    if is_gpu_available and is_paddle_gpu_version:
        logging.info("==> SUCESSO: Ambiente validado. O dispositivo 'CUDA' será utilizado.")
        return "cuda"
    else:
        logging.warning("==> AVISO: Condições para usar CUDA não atendidas. Revertendo para 'CPU'.")
        return "cpu"

# --- Dataclass para Agrupar Configurações ---
# Usar uma dataclass organiza todas as variáveis em um único objeto.
@dataclass(frozen=True)
class AppSettings:
    """Agrupa todas as configurações da aplicação de forma imutável."""
    # Caminhos
    CONFIG_FILE: Path = field(default_factory=lambda: ROOT_DIR / os.getenv("CONFIG_FILE", "config.json"))
    LOGS_SAVE_DIR: Path = field(default_factory=lambda: ROOT_DIR / os.getenv("LOGS_SAVE_DIR", "log/"))

    DB_CONNECTION: Path = field(default_factory=lambda: ROOT_DIR / os.getenv("DB_CONNECTION", "db/captures.db"))
    CAPTURES_SAVE_DIR: Path = field(default_factory=lambda: get_env_path(ROOT_DIR, "CAPTURES_SAVE_DIR"))
    SUSPECT_DETECTIONS_SAVE_DIR: Path = field(default_factory=lambda: get_env_path(ROOT_DIR, "SUSPECT_DETECTIONS_SAVE_DIR"))

    # Modelos
    BASE_PLATE_MODEL: Path = field(default_factory=lambda: get_env_path(ROOT_DIR, "BASE_PLATE_MODEL"))
    OCR_CHAR_DICT_FILE: Path = field(default_factory=lambda: get_env_path(ROOT_DIR, "OCR_CHAR_DICT_FILE"))
    CHAR_CORRECTIONS_FILE: Path = field(default_factory=lambda: get_env_path(ROOT_DIR, "CHAR_CORRECTIONS_FILE"))
    
    # Seleção de Backend e Dispositivo
    OCR_BACKEND: str = field(default_factory=lambda: os.getenv("OCR_BACKEND", "paddle").lower())
    COMPUTE_DEVICE: str = field(default_factory=_determine_compute_device)

    # Flags de Controle
    USE_OCR_DETECTION: bool = field(default_factory=lambda: get_env_bool("USE_OCR_DETECTION"))
    USE_OCR_ANGLE_CLS: bool = field(default_factory=lambda: get_env_bool("USE_OCR_ANGLE_CLS"))
    USE_CONTINUOUS_TRIES: bool = field(default_factory=lambda: get_env_bool("USE_CONTINUOUS_TRIES", False))
    SAVE_SUSPECT_DETECTIONS: bool = field(default_factory=lambda: get_env_bool("SAVE_SUSPECT_DETECTIONS"))
    SHOW_CAPTURES: bool = field(default_factory=lambda: get_env_bool("SHOW_CAPTURES"))
    CALCULATE_FPS: bool = field(default_factory=lambda: get_env_bool("CALCULATE_FPS", False))
    CROP_MARGIN: int = field(default_factory=lambda: get_env_int("CROP_MARGIN"))

    # Outros
    API_ENDPOINT: str | None = field(default_factory=lambda: os.getenv("API_ENDPOINT"))
    READING_FORMATS: list[str] = field(default_factory=lambda: os.getenv("READING_FORMATS", "").split(","))

    # Carregados de arquivos
    INPUT_SOURCES: dict = field(init=False)
    CHAR_CORRECTIONS: dict = field(init=False)

    # Modelos de OCR (lógica de seleção)
    OCR_DETECTION_MODEL: Path | None = field(init=False)
    OCR_RECOGNITION_MODEL: Path | None = field(init=False)
    OCR_CLASSIFICATION_MODEL: Path | None = field(init=False)

    def __post_init__(self):
        """Carrega configurações que dependem de outras ou de arquivos."""
        # Carrega dados do config.json
        with open(self.CONFIG_FILE) as f:
            config_data = json.load(f)
        # Usamos object.__setattr__ para modificar o objeto "frozen"
        object.__setattr__(self, 'INPUT_SOURCES', config_data.get("input_sources", {}))

        # Carrega correções de caracteres
        with open(self.CHAR_CORRECTIONS_FILE, 'r', encoding='utf-8') as f:
            object.__setattr__(self, 'CHAR_CORRECTIONS', json.load(f))

        # Lógica de seleção dos modelos de OCR
        if self.OCR_BACKEND == 'openvino':
            det_model = get_env_path(ROOT_DIR, "OPENVINO_OCR_DET_MODEL")
            rec_model = get_env_path(ROOT_DIR, "OPENVINO_OCR_REC_MODEL")
        else: # Padrão 'paddle'
            det_model = get_env_path(ROOT_DIR, "PADDLE_OCR_DET_MODEL")
            rec_model = get_env_path(ROOT_DIR, "PADDLE_OCR_REC_MODEL")

        object.__setattr__(self, 'OCR_RECOGNITION_MODEL', rec_model)
        object.__setattr__(self, 'OCR_DETECTION_MODEL', det_model if self.USE_OCR_DETECTION else None)
        
        cls_model = get_env_path(ROOT_DIR, "OCR_CLASSIFICATION_MODEL") if self.USE_OCR_ANGLE_CLS else None
        object.__setattr__(self, 'OCR_CLASSIFICATION_MODEL', cls_model)

        # Cria diretórios necessários
        self.LOGS_SAVE_DIR.mkdir(exist_ok=True)
        self.DB_CONNECTION.parent.mkdir(exist_ok=True)


# --- Instância Única de Configurações ---
# Esta instância será importada por outros módulos.
settings = AppSettings()