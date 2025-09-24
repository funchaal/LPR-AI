import logging
import os
import openvino.runtime as ov

def optimize_ocr_model(model_dir, device='cpu', output_base_dir="app/models/ocr"):
    """
    Verifica se existe modelo OCR otimizado baseado no dispositivo de destino.
    Para CPU: verifica se existe modelo OpenVINO (.xml)
    Para GPU: verifica se existe modelo TensorRT (.engine)
    Para outros: usa modelo PaddlePaddle original
    
    Args:
        model_dir (str): Caminho para o modelo PaddlePaddle original (modelo base)
        device (str): Dispositivo de destino ('cpu', 'gpu', etc.)
        output_base_dir (str): Diretório base para modelos convertidos
        
    Returns:
        str: Caminho para o modelo otimizado se existir e válido, senão modelo base
    """
    # Sempre retorna o modelo base se não foi fornecido
    if not model_dir:
        logging.warning("Nenhum modelo base fornecido")
        return model_dir
        
    # Se o modelo base não existir, retorna None ou caminho inválido
    if not os.path.exists(model_dir):
        logging.error(f"Modelo base não encontrado: {model_dir}")
        return model_dir
        
    # Se for GPU, verifica se existe modelo TensorRT (.engine)
    if device == 'gpu':
        try:
            model_name = os.path.basename(model_dir)
            
            # Verifica se já existe versão TensorRT
            tensorrt_path = os.path.join(output_base_dir, "tensorrt", model_name, f"{model_name}.engine")
            
            if os.path.exists(tensorrt_path):
                try:
                    # Verifica se o arquivo não está vazio
                    if os.path.getsize(tensorrt_path) <= 0:
                        logging.warning(f"Modelo TensorRT existe mas está vazio, usando modelo base: {model_dir}")
                        return model_dir
                    
                    logging.info(f"Modelo TensorRT encontrado: {tensorrt_path}")
                    return tensorrt_path
                    
                except Exception as validation_error:
                    logging.warning(f"Erro na validação do modelo TensorRT: {validation_error}, usando modelo base: {model_dir}")
                    return model_dir
            else:
                # Modelo TensorRT não existe, usa o modelo base
                logging.info(f"Modelo TensorRT não encontrado para {model_name}, usando modelo base: {model_dir}")
                return model_dir
                
        except Exception as e:
            logging.error(f"Erro ao verificar modelo TensorRT para {model_dir}: {e}, usando modelo base")
            return model_dir
    
    # Se for outro dispositivo, usa o modelo PaddlePaddle original (modelo base)
    if device != 'cpu':
        logging.info(f"Dispositivo {device} detectado, usando modelo PaddlePaddle base: {model_dir}")
        return model_dir
        
    try:
        model_name = os.path.basename(model_dir)
        
        # Verifica se já existe versão OpenVINO otimizada
        openvino_path = os.path.join(model_dir.replace('paddlepaddle', 'openvino'), f"{model_name}.xml")
        
        if os.path.exists(openvino_path):
            try:
                # Verifica se o arquivo não está vazio
                if os.path.getsize(openvino_path) <= 0:
                    logging.warning(f"Modelo OpenVINO existe mas está vazio, usando modelo base: {model_dir}")
                    return model_dir
                
                # Teste básico para verificar se o modelo OpenVINO é válido
                test_model = ov.Core().read_model(openvino_path)
                if test_model:
                    logging.info(f"Modelo OpenVINO encontrado e validado: {openvino_path}")
                    return openvino_path
                else:
                    logging.warning(f"Modelo OpenVINO inválido, usando modelo base: {model_dir}")
                    return model_dir
                    
            except Exception as validation_error:
                logging.warning(f"Erro na validação do modelo OpenVINO: {validation_error}, usando modelo base: {model_dir}")
                return model_dir
        else:
            # Modelo OpenVINO não existe, usa o modelo base
            logging.info(f"Modelo OpenVINO não encontrado para {model_name} ({openvino_path}), usando modelo base: {model_dir}")
            return model_dir
            
    except Exception as e:
        logging.error(f"Erro ao verificar modelo otimizado para {model_dir}: {e}, usando modelo base")
        return model_dir