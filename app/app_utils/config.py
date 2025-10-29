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
import sys
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
    # Seção 1: Configurações Gerais e de Debug
    INSTANCE_ID: str | None = field(default_factory=lambda: os.getenv("INSTANCE_ID"))
    CONFIG_FILE: Path = field(default_factory=lambda: ROOT_DIR / os.getenv("CONFIG_FILE", "config.json"))
    DEBUG: bool = field(default_factory=lambda: get_env_bool("DEBUG", False))
    SHOW_CAPTURES: bool = field(default_factory=lambda: get_env_bool("SHOW_CAPTURES"))
    CALCULATE_FPS: bool = field(default_factory=lambda: get_env_bool("CALCULATE_FPS", False))
    FPS_LOG_LIMIAR: int = field(default_factory=lambda: get_env_int("FPS_LOG_LIMIAR", 5))

    # Seção 2: Dispositivos de Inferência (IA)
    PLATE_DETECTION_DEVICE: str | None = field(default_factory=lambda: os.getenv("PLATE_DETECTION_DEVICE"))
    OCR_DEVICE: str | None = field(default_factory=lambda: os.getenv("OCR_DEVICE"))

    # Seção 3: Caminhos de Arquivos e Modelos (Internos da Imagem)
    PLATE_DETECTION_MODEL: Path = field(default_factory=lambda: get_env_path(ROOT_DIR, "PLATE_DETECTION_MODEL"))
    # Corresponde a OCR_DET_MODEL no .env
    OCR_DETECTION_MODEL: Path | None = field(default_factory=lambda: get_env_path(ROOT_DIR, "OCR_DET_MODEL"))
    # Corresponde a OCR_REC_MODEL no .env
    OCR_RECOGNITION_MODEL: Path | None = field(default_factory=lambda: get_env_path(ROOT_DIR, "OCR_REC_MODEL"))
    OCR_CHAR_DICT_FILE: Path = field(default_factory=lambda: get_env_path(ROOT_DIR, "OCR_CHAR_DICT_FILE"))
    CHAR_CORRECTIONS_FILE: Path = field(default_factory=lambda: get_env_path(ROOT_DIR, "CHAR_CORRECTIONS_FILE"))

    # Seção 4: Configurações de Processamento e Lógica do Algoritmo
    USE_OCR_DETECTION: bool = field(default_factory=lambda: get_env_bool("USE_OCR_DETECTION"))
    MIN_PLATE_DETECTION_HEIGHT: int = field(default_factory=lambda: get_env_int("MIN_PLATE_DETECTION_HEIGHT", 16))
    MIN_PLATE_DETECTION_WIDTH: int = field(default_factory=lambda: get_env_int("MIN_PLATE_DETECTION_WIDTH", 10))
    OCR_TARGET_HEIGHT: int = field(default_factory=lambda: get_env_int("OCR_TARGET_HEIGHT", 48))
    OCR_TARGET_WIDTH: int = field(default_factory=lambda: get_env_int("OCR_TARGET_WIDTH", 160))
    CROP_MARGIN: int = field(default_factory=lambda: get_env_int("CROP_MARGIN"))
    READING_FORMATS: list[str] = field(default_factory=lambda: os.getenv("READING_FORMATS", "").split(","))
    READINGS_FILTER_REGEX: str | None = field(default_factory=lambda: os.getenv("READINGS_FILTER_REGEX"))
    STABILITY_MAX_COORDINATE_DIFFERENCE: int = field(default_factory=lambda: get_env_int("STABILITY_MAX_COORDINATE_DIFFERENCE"))
    STATIONARY_FRAME_THRESHOLD: int = field(default_factory=lambda: get_env_int("STATIONARY_FRAME_THRESHOLD"))
    MAX_NO_FRAME_COUNT: int = field(default_factory=lambda: get_env_int("MAX_NO_FRAME_COUNT", 10))
    USE_CONTINUOUS_TRIES: bool = field(default_factory=lambda: get_env_bool("USE_CONTINUOUS_TRIES", False))

    # Seção 5: Armazenamento, Logs e Banco de Dados (Apontando para o Volume)
    LOGS_SAVE_DIR: Path = field(default_factory=lambda: ROOT_DIR / os.getenv("LOGS_SAVE_DIR", "log/"))
    LOGS_SAVE_DAYS: int = field(default_factory=lambda: get_env_int("LOGS_SAVE_DAYS", 30))
    SAVE_DB: bool = field(default_factory=lambda: get_env_bool("SAVE_DB"))
    DB_CONNECTION: Path = field(default_factory=lambda: ROOT_DIR / os.getenv("DB_CONNECTION", "db/captures.db"))
    SAVE_CAPTURES: bool = field(default_factory=lambda: get_env_bool("SAVE_CAPTURES"))
    CAPTURES_SAVE_DIR: Path = field(default_factory=lambda: get_env_path(ROOT_DIR, "CAPTURES_SAVE_DIR"))
    SAVE_SUSPECT_DETECTIONS: bool = field(default_factory=lambda: get_env_bool("SAVE_SUSPECT_DETECTIONS"))
    SUSPECT_DETECTIONS_SAVE_DIR: Path = field(default_factory=lambda: get_env_path(ROOT_DIR, "SUSPECT_DETECTIONS_SAVE_DIR"))

    # Seção 6: Integração com API Externa (Opcional)
    API_ENDPOINT: str | None = field(default_factory=lambda: os.getenv("API_ENDPOINT"))
    API_USER: str | None = field(default_factory=lambda: os.getenv("API_USER"))
    API_PASSWORD: str | None = field(default_factory=lambda: os.getenv("API_PASSWORD"))
    CLOSE_API_ENDPOINT: str | None = field(default_factory=lambda: os.getenv("CLOSE_API_ENDPOINT"))

    # --- Campos Carregados Dinamicamente (init=False) ---
    # Estes campos são inicializados no método __post_init__.
    INPUT_SOURCES: dict = field(init=False)
    CHAR_CORRECTIONS_DICT: dict = field(init=False)

    def __post_init__(self):
        """Executa após a inicialização do dataclass para carregar configurações complexas."""
        # O uso de `object.__setattr__` é necessário porque o dataclass é imutável (frozen=True).
        # Este método permite atribuir valores aos campos `init=False` após a criação da instância.

        # Carrega as fontes de entrada do arquivo `config.json`.
        try:
            with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except FileNotFoundError:
            print(f"Arquivo de configuração '{self.CONFIG_FILE}' não encontrado.", file=sys.stderr)
            config_data = {}
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Erro ao decodificar o JSON do arquivo '{self.CONFIG_FILE}'.", e.doc, e.pos)

        # Processa e valida cada fonte de entrada do `config.json`.
        validated_sources = {}
        sources_from_config = config_data.get("input_sources", {})
        if not sources_from_config:
            print("Nenhuma 'input_sources' encontrada no config.json!", file=sys.stderr)
        
        for name, data in sources_from_config.items():
            # TODO: Adicionar lógica de validação para cada fonte (ex: verificar se o endpoint é acessível).
            updated_data = data.copy()
            validated_sources[name] = updated_data
        
        object.__setattr__(self, 'INPUT_SOURCES', validated_sources)

        # Carrega o dicionário de correções de caracteres.

        def load_char_corrections() -> dict:
            """Carrega o dicionário de correções de caracteres a partir de um arquivo JSON."""
            corrections_file = self.CHAR_CORRECTIONS_FILE
            if not corrections_file.exists():
                print(f"Arquivo de correções de caracteres '{corrections_file}' não encontrado.", file=sys.stderr)
                return {}

            with open(corrections_file, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(f"Erro ao decodificar o JSON do arquivo '{corrections_file}'.", e.doc, e.pos)

        object.__setattr__(self, 'CHAR_CORRECTIONS_DICT', load_char_corrections())

# --- INSTÂNCIA ÚNICA DE CONFIGURAÇÕES ---
# Cria uma instância única e imutável das configurações que será importada
# por outros módulos da aplicação.
settings = AppSettings()
