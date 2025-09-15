# app_utils/model_optimizer.py

import logging
import shutil
from pathlib import Path
from ultralytics import YOLO

# Ponto 3: Usando import relativo para robustez, buscando o config.py no diretório pai (app/)
from app_utils.config import settings, APP_DIR

def ensure_best_model(device) -> Path:
    """
    Verifica o ambiente e garante que o melhor modelo (otimizado ou base) esteja pronto para uso.
    """
    base_model_path = settings.BASE_PLATE_MODEL

    if not base_model_path or not base_model_path.exists():
        logging.critical(f"Modelo base não encontrado: '{base_model_path}'!")
        raise FileNotFoundError(f"Modelo base não foi encontrado em '{base_model_path}'")

    model_stem = base_model_path.stem

    # --- LÓGICA DE SELEÇÃO DE BACKEND ---

    # CASO 1: Dispositivo é CUDA, otimizar para TensorRT
    if 'gpu' in device:
        engine_dir = APP_DIR / 'models' / 'plate' / 'engine'
        engine_path = engine_dir / (model_stem + '.engine')
        engine_dir.mkdir(parents=True, exist_ok=True)

        if engine_path.exists() and base_model_path.stat().st_mtime <= engine_path.stat().st_mtime:
            logging.info(f"Usando modelo TensorRT existente e atualizado: '{engine_path}'")
            return engine_path
        
        logging.warning(f"Gerando novo modelo TensorRT para '{base_model_path.name}'...")
        try:
            model = YOLO(base_model_path)
            exported_file = model.export(format='engine', half=True, device=0)
            shutil.move(str(exported_file), engine_path)
            logging.info(f"Modelo TensorRT salvo com sucesso em '{engine_path}'")
            return engine_path
        except Exception as e:
            logging.error(f"Falha ao exportar para TensorRT: {e}. Usando modelo .pt padrão.")
            return base_model_path

    # CASO 2: Dispositivo é CPU, otimizar para OpenVINO
    elif device == 'cpu':
        # Ponto 1: Usa o nome PADRÃO que a Ultralytics gera (ex: nome_openvino_model)
        # Isso é mais seguro e evita quebrar referências internas do modelo.
        ov_model_dir_name = model_stem + '_openvino_model'
        ov_target_dir = APP_DIR / 'models' / 'plate' / 'openvino' / ov_model_dir_name
        ov_xml_path = ov_target_dir / (model_stem + '.xml')
        
        ov_target_dir.parent.mkdir(parents=True, exist_ok=True)

        if ov_target_dir.exists() and ov_xml_path.exists() and base_model_path.stat().st_mtime <= ov_xml_path.stat().st_mtime:
            logging.info(f"Usando modelo OpenVINO existente e atualizado: '{ov_target_dir}'")
            return ov_target_dir

        logging.warning(f"Gerando novo modelo OpenVINO para '{base_model_path.name}'...")
        try:
            if ov_target_dir.exists():
                shutil.rmtree(ov_target_dir)
            
            model = YOLO(base_model_path)
            # Ponto 2: Usando half=True, conforme seu script de teste que funcionou.
            temp_exported_dir = model.export(format='openvino', half=True)
            
            shutil.move(str(temp_exported_dir), ov_target_dir)
            logging.info(f"Modelo OpenVINO salvo com sucesso em '{ov_target_dir}'")
            return ov_target_dir
        except Exception as e:
            logging.error(f"Falha ao exportar para OpenVINO: {e}. Usando modelo .pt padrão.")
            return base_model_path

    # CASO 3: Fallback
    else:
        logging.info(f"Nenhuma otimização para '{device}'. Usando .pt padrão.")
        return base_model_path