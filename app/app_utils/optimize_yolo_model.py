# app_utils/model_optimizer.py

import logging
import os
from pathlib import Path

from app_utils.yolo2tensorrt import yolo2tensorrt
from app_utils.yolo2onnx import yolo2onnx

# Ponto 3: Usando import relativo para robustez, buscando o config.py no diretório pai (app/)
from app_utils.config import settings, APP_DIR

def optimize_yolo_model(base_model_path, device, yolo_inference_device) -> Path:
    """
    Verifica se existe modelo YOLO otimizado baseado no dispositivo de destino.
    Para GPU: verifica se existe modelo TensorRT (.engine) ou tenta criar usando yolo2tensorrt
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
        # Tenta TensorRT primeiro
        try:
            engine_dir = APP_DIR / 'models' / 'plate' / 'engine'
            engine_path = engine_dir / (model_stem + '.engine')
            
            if engine_path.exists():
                engine_size = engine_path.stat().st_size
                if engine_size > 0:
                    logging.info(f"Modelo TensorRT encontrado e validado: '{engine_path}'")
                    return engine_path
                else:
                    logging.warning(f"Modelo TensorRT existe mas está vazio, tentando recriar...")
                    engine_path.unlink(missing_ok=True)
            
            logging.info(f"Tentando criar modelo TensorRT para {model_stem}...")
            created_engine_path = yolo2tensorrt(base_model_path, yolo_inference_device)
            
            if created_engine_path and created_engine_path.exists():
                created_size = created_engine_path.stat().st_size
                if created_size > 0:
                    logging.info(f"Modelo TensorRT criado com sucesso: '{created_engine_path}'")
                    return created_engine_path
                else:
                    logging.warning(f"Modelo TensorRT criado mas está vazio.")
            else:
                logging.warning(f"Falha ao criar modelo TensorRT.")

        except Exception as tensorrt_error:
            logging.error(f"Erro ao processar modelo TensorRT: {tensorrt_error}")

        # Fallback para ONNX se TensorRT falhar
        logging.info("Fallback para modelo ONNX...")
        try:
            onnx_dir = APP_DIR / 'models' / 'plate' / 'onnx'
            onnx_path = onnx_dir / (model_stem + '.onnx')

            if onnx_path.exists():
                onnx_size = onnx_path.stat().st_size
                if onnx_size > 0:
                    logging.info(f"Modelo ONNX encontrado e validado: '{onnx_path}'")
                    return onnx_path
                else:
                    logging.warning(f"Modelo ONNX existe mas está vazio, tentando recriar...")
                    onnx_path.unlink(missing_ok=True)

            logging.info(f"Tentando criar modelo ONNX para {model_stem}...")
            created_onnx_path = yolo2onnx(base_model_path, yolo_inference_device)

            if created_onnx_path and created_onnx_path.exists():
                created_size = created_onnx_path.stat().st_size
                if created_size > 0:
                    logging.info(f"Modelo ONNX criado com sucesso: '{created_onnx_path}'")
                    return created_onnx_path
                else:
                    logging.warning(f"Modelo ONNX criado mas está vazio.")
            else:
                logging.warning(f"Falha ao criar modelo ONNX.")

        except Exception as onnx_error:
            logging.error(f"Erro ao processar modelo ONNX: {onnx_error}")

        # Fallback final para o modelo base
        logging.warning(f"Usando modelo base: {base_model_path}")
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