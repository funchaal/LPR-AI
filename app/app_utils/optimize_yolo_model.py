# app_utils/model_optimizer.py

import logging
import os
from pathlib import Path

# Ponto 3: Usando import relativo para robustez, buscando o config.py no diretório pai (app/)
from app_utils.config import settings, APP_DIR

def optimize_yolo_model(base_model_path, device) -> Path:
    """
    Verifica se existe modelo YOLO otimizado baseado no dispositivo de destino.
    Para GPU: verifica se existe modelo TensorRT (.engine)
    Para CPU: verifica se existe modelo OpenVINO (.xml)
    Para outros: usa modelo YOLO original (.pt)
    
    Args:
        base_model_path (Path): Caminho para o modelo YOLO original (modelo base)
        device (str): Dispositivo de destino ('cpu', 'gpu', etc.)
        
    Returns:
        Path: Caminho para o modelo otimizado se existir e válido, senão modelo base
    """
    
    if not base_model_path or not base_model_path.exists():
        logging.error(f"Modelo base não encontrado: '{base_model_path}'!")
        raise FileNotFoundError(f"Modelo base não foi encontrado em '{base_model_path}'")
    
    model_stem = base_model_path.stem
    
    # --- LÓGICA DE SELEÇÃO DE BACKEND ---
    
    # CASO 1: Dispositivo é GPU, verifica modelo TensorRT
    if 'gpu' in device:
        try:
            engine_dir = APP_DIR / 'models' / 'plate' / 'engine'
            engine_path = engine_dir / (model_stem + '.engine')
            
            # Verifica se já existe modelo TensorRT
            if engine_path.exists():
                try:
                    # Teste básico de integridade
                    engine_size = engine_path.stat().st_size
                    if engine_size > 0:
                        logging.info(f"Modelo TensorRT encontrado e validado: '{engine_path}'")
                        return engine_path
                    else:
                        logging.warning(f"Modelo TensorRT existe mas está vazio, usando modelo base: {base_model_path}")
                        return base_model_path
                except Exception as validation_error:
                    logging.warning(f"Erro na validação do modelo TensorRT: {validation_error}, usando modelo base: {base_model_path}")
                    return base_model_path
            else:
                # Modelo TensorRT não existe, usa o modelo base
                logging.info(f"Modelo TensorRT não encontrado para {model_stem}, usando modelo base: {base_model_path}")
                return base_model_path
                
        except Exception as e:
            logging.error(f"Erro ao verificar modelo TensorRT para {base_model_path}: {e}, usando modelo base")
            return base_model_path
    
    # CASO 2: Dispositivo é CPU, verifica modelo OpenVINO
    elif device == 'cpu':
        try:
            ov_model_dir_name = model_stem + '_openvino_model'
            ov_target_dir = APP_DIR / 'models' / 'plate' / 'openvino' / ov_model_dir_name
            ov_xml_path = ov_target_dir / (model_stem + '.xml')
            
            # Verifica se já existe modelo OpenVINO
            if ov_target_dir.exists() and ov_xml_path.exists():
                try:
                    # Teste básico de integridade
                    xml_size = ov_xml_path.stat().st_size
                    if xml_size > 0:
                        logging.info(f"Modelo OpenVINO encontrado e validado: '{ov_target_dir}'")
                        return ov_target_dir
                    else:
                        logging.warning(f"Modelo OpenVINO existe mas está corrompido, usando modelo base: {base_model_path}")
                        return base_model_path
                except Exception as validation_error:
                    logging.warning(f"Erro na validação do modelo OpenVINO: {validation_error}, usando modelo base: {base_model_path}")
                    return base_model_path
            else:
                # Modelo OpenVINO não existe, usa o modelo base
                logging.info(f"Modelo OpenVINO não encontrado para {model_stem}, usando modelo base: {base_model_path}")
                return base_model_path
                
        except Exception as e:
            logging.error(f"Erro ao verificar modelo OpenVINO para {base_model_path}: {e}, usando modelo base")
            return base_model_path
    
    # CASO 3: Outros dispositivos, usa modelo base
    else:
        logging.info(f"Dispositivo '{device}' detectado, usando modelo base: {base_model_path}")
        return base_model_path