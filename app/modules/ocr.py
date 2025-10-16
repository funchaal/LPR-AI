# modules/ocr.py
# Módulo "roteador" que inicializa o backend de OCR apropriado (OpenVINO ou ONNX).

import logging
import os

def init_ocr(det_model_dir, rec_model_dir, use_det=True, device='cpu', char_dict_file=None, **kwargs):
    """
    Inicializa o motor de OCR, selecionando o melhor backend disponível.
    - Para GPU, usa ONNX Runtime com CUDA.
    - Para CPU, tenta OpenVINO e, se falhar, usa ONNX Runtime com CPU.
    """
    # setup_logger() # Descomente se você tiver esta função
    logging.info("--- Iniciando seleção de backend de OCR ---")

    # Se o dispositivo for GPU, a escolha é direta: ONNX com CUDA.
    if 'gpu' in device:
        logging.info("Dispositivo GPU detectado. Usando backend ONNX com CUDA.")
        try:
            from modules.onnx_ocr import init_onnx_ocr
            return init_onnx_ocr(
                det_model_dir=det_model_dir,
                rec_model_dir=rec_model_dir,
                use_det=use_det,
                device=device,
                char_dict_file=char_dict_file,
                **kwargs
            )
        except Exception as e:
            logging.critical(f"Falha ao inicializar o backend ONNX para GPU: {e}", exc_info=True)
            raise RuntimeError("Não foi possível inicializar o serviço de OCR para GPU.") from e

    # Se o dispositivo for CPU, tentamos a otimização com OpenVINO primeiro.
    elif 'cpu' in device:
        # Tentativa 1: OpenVINO
        try:
            logging.info("Dispositivo CPU detectado. Tentando backend otimizado OpenVINO...")
            # Supondo que você tenha um módulo openvino_ocr com uma função de inicialização
            from modules.openvino_ocr import init_openvino_ocr 
            
            # A função init_openvino_ocr deve ser responsável por encontrar/converter os modelos
            return init_openvino_ocr(
                det_model_dir=det_model_dir,
                rec_model_dir=rec_model_dir,
                use_det=use_det,
                device=device,
                char_dict_file=char_dict_file,
                **kwargs
            )
        except Exception as e:
            logging.warning(f"Backend OpenVINO não pôde ser inicializado. Erro: {e}")
            logging.info("Fallback para o backend ONNX com CPUExecutionProvider.")
            
            # Tentativa 2: Fallback para ONNX com CPU
            try:
                from modules.onnx_ocr import init_onnx_ocr
                return init_onnx_ocr(
                    det_model_dir=det_model_dir,
                    rec_model_dir=rec_model_dir,
                    use_det=use_det,
                    device=device, # Passando 'cpu'
                    char_dict_file=char_dict_file,
                    **kwargs
                )
            except Exception as e_onnx:
                logging.critical(f"Falha ao inicializar o backend de fallback ONNX para CPU: {e_onnx}", exc_info=True)
                raise RuntimeError("Não foi possível inicializar nenhum serviço de OCR para CPU.") from e_onnx
    
    else:
        raise ValueError(f"Dispositivo '{device}' não é suportado.")