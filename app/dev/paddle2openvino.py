import logging
import os
from pathlib import Path
import subprocess
import openvino.runtime as ov

def paddle2openvino_alternative(paddle_model_dir, output_base_dir="app/models/ocr"):
    """
    Método alternativo para conversão quando paddlex não está disponível.
    Tenta usar paddle2onnx diretamente e depois converte para OpenVINO.
    
    Args:
        paddle_model_dir (str): Caminho para o diretório do modelo PaddlePaddle
        output_base_dir (str): Diretório base onde salvar os modelos convertidos
        
    Returns:
        str or None: Caminho para o arquivo .xml do modelo OpenVINO ou None se falhou
    """
    if not paddle_model_dir or not os.path.exists(paddle_model_dir):
        logging.warning(f"Diretório do modelo PaddlePaddle não existe: {paddle_model_dir}")
        return None
        
    try:
        model_name = os.path.basename(paddle_model_dir)
        
        # Cria diretórios de saída
        onnx_dir = os.path.join(output_base_dir, "onnx", model_name)
        openvino_dir = os.path.join(output_base_dir, "openvino", model_name)
        
        Path(onnx_dir).mkdir(parents=True, exist_ok=True)
        Path(openvino_dir).mkdir(parents=True, exist_ok=True)
        
        # Passo 1: PaddlePaddle -> ONNX usando paddle2onnx diretamente
        onnx_model_path = os.path.join(onnx_dir, "inference.onnx")
        
        logging.info(f"Convertendo PaddlePaddle para ONNX (método alternativo): {model_name}")
        
        try:
            import paddle2onnx
            
            # Procura pelos arquivos do modelo PaddlePaddle
            pdmodel_path = None
            pdiparams_path = None
            
            for file in os.listdir(paddle_model_dir):
                if file.endswith('.pdmodel'):
                    pdmodel_path = os.path.join(paddle_model_dir, file)
                elif file.endswith('.pdiparams'):
                    pdiparams_path = os.path.join(paddle_model_dir, file)
            
            if not pdmodel_path or not pdiparams_path:
                logging.error(f"Arquivos .pdmodel ou .pdiparams não encontrados em {paddle_model_dir}")
                return None
            
            # Converte usando paddle2onnx diretamente
            onnx_model = paddle2onnx.command.c_paddle_to_onnx(
                model_file=pdmodel_path,
                params_file=pdiparams_path,
                opset_version=11,
                enable_onnx_checker=True
            )
            
            with open(onnx_model_path, "wb") as f:
                f.write(onnx_model)
            
            if not os.path.exists(onnx_model_path) or os.path.getsize(onnx_model_path) == 0:
                logging.error(f"Falha na geração do modelo ONNX: {onnx_model_path}")
                return None
                
            logging.info(f"Conversão Paddle->ONNX (alternativa) concluída: {onnx_model_path}")
            
        except ImportError:
            logging.error("paddle2onnx não está disponível para método alternativo")
            return None
        except Exception as paddle2onnx_error:
            logging.error(f"Erro na conversão paddle2onnx: {paddle2onnx_error}")
            return None
        
        # Passo 2: ONNX -> OpenVINO (mesmo processo)
        openvino_xml_path = os.path.join(openvino_dir, f"{model_name}.xml")
        
        logging.info(f"Convertendo ONNX para OpenVINO: {model_name}")
        
        try:
            ov_model = ov.convert_model(onnx_model_path)
            ov.save_model(ov_model, openvino_xml_path)
        except Exception as ov_error:
            logging.error(f"Erro específico na conversão OpenVINO: {ov_error}")
            return None
        
        if not os.path.exists(openvino_xml_path):
            logging.error(f"Arquivo OpenVINO não foi gerado: {openvino_xml_path}")
            return None
            
        # Verifica se os arquivos .xml e .bin foram criados
        openvino_bin_path = openvino_xml_path.replace('.xml', '.bin')
        if not os.path.exists(openvino_bin_path):
            logging.warning(f"Arquivo .bin não encontrado: {openvino_bin_path}")
            
        # Verifica tamanhos dos arquivos
        xml_size = os.path.getsize(openvino_xml_path)
        bin_size = os.path.getsize(openvino_bin_path) if os.path.exists(openvino_bin_path) else 0
        
        if xml_size == 0:
            logging.error(f"Arquivo .xml está vazio: {openvino_xml_path}")
            return None
            
        logging.info(f"Conversão alternativa concluída: {openvino_xml_path} (XML: {xml_size} bytes, BIN: {bin_size} bytes)")
        
        # Remove arquivo ONNX temporário
        try:
            os.remove(onnx_model_path)
            logging.debug(f"Arquivo ONNX temporário removido: {onnx_model_path}")
        except Exception as cleanup_error:
            logging.warning(f"Erro ao remover arquivo ONNX temporário: {cleanup_error}")
        
        return openvino_xml_path
        
    except Exception as e:
        logging.error(f"Erro na conversão alternativa para {model_name}: {e}")
        return None

def paddle2openvino(paddle_model_dir, output_base_dir="app/models/ocr"):
    """
    Converte modelo PaddlePaddle para OpenVINO via ONNX.
    
    Args:
        paddle_model_dir (str): Caminho para o diretório do modelo PaddlePaddle
        output_base_dir (str): Diretório base onde salvar os modelos convertidos
        
    Returns:
        str or None: Caminho para o arquivo .xml do modelo OpenVINO ou None se falhou
    """
    if not paddle_model_dir or not os.path.exists(paddle_model_dir):
        logging.warning(f"Diretório do modelo PaddlePaddle não existe: {paddle_model_dir}")
        return None
        
    try:
        model_name = os.path.basename(paddle_model_dir)
        
        # Cria diretórios de saída
        onnx_dir = os.path.join(output_base_dir, "onnx", model_name)
        openvino_dir = os.path.join(output_base_dir, "openvino", model_name)
        
        Path(onnx_dir).mkdir(parents=True, exist_ok=True)
        Path(openvino_dir).mkdir(parents=True, exist_ok=True)
        
        # Passo 1: PaddlePaddle -> ONNX
        onnx_model_path = os.path.join(onnx_dir, "inference.onnx")
        
        logging.info(f"Convertendo PaddlePaddle para ONNX: {model_name}")
        
        # Verifica se paddlex está disponível
        try:
            result_check = subprocess.run(
                ["paddlex", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result_check.returncode != 0:
                logging.error("paddlex não está disponível no sistema")
                return None
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logging.error(f"paddlex não encontrado: {e}")
            return None
        
        # Comando paddlex para conversão paddle -> onnx
        paddle_cmd = [
            "paddlex", "--paddle2onnx",
            "paddle_model_dir", paddle_model_dir,
            "onnx_model_dir", onnx_model_path
        ]
        
        result = subprocess.run(
            paddle_cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutos timeout
            cwd=os.getcwd()  # Define diretório de trabalho
        )
        
        if result.returncode != 0:
            logging.error(f"Erro na conversão Paddle->ONNX (code {result.returncode}): {result.stderr}")
            if result.stdout:
                logging.info(f"stdout: {result.stdout}")
            return None
            
        if not os.path.exists(onnx_model_path):
            logging.error(f"Arquivo ONNX não foi gerado: {onnx_model_path}")
            return None
            
        # Verifica se o arquivo ONNX tem tamanho válido
        if os.path.getsize(onnx_model_path) == 0:
            logging.error(f"Arquivo ONNX está vazio: {onnx_model_path}")
            return None
            
        logging.info(f"Conversão Paddle->ONNX concluída: {onnx_model_path}")
        
        # Passo 2: ONNX -> OpenVINO
        openvino_xml_path = os.path.join(openvino_dir, f"{model_name}.xml")
        
        logging.info(f"Convertendo ONNX para OpenVINO: {model_name}")
        
        # Converte ONNX para OpenVINO
        try:
            ov_model = ov.convert_model(onnx_model_path)
            ov.save_model(ov_model, openvino_xml_path)
        except Exception as ov_error:
            logging.error(f"Erro específico na conversão OpenVINO: {ov_error}")
            return None
        
        if not os.path.exists(openvino_xml_path):
            logging.error(f"Arquivo OpenVINO não foi gerado: {openvino_xml_path}")
            return None
            
        # Verifica se os arquivos .xml e .bin foram criados
        openvino_bin_path = openvino_xml_path.replace('.xml', '.bin')
        if not os.path.exists(openvino_bin_path):
            logging.warning(f"Arquivo .bin não encontrado: {openvino_bin_path}")
            
        # Verifica tamanhos dos arquivos
        xml_size = os.path.getsize(openvino_xml_path)
        bin_size = os.path.getsize(openvino_bin_path) if os.path.exists(openvino_bin_path) else 0
        
        if xml_size == 0:
            logging.error(f"Arquivo .xml está vazio: {openvino_xml_path}")
            return None
            
        logging.info(f"Conversão ONNX->OpenVINO concluída: {openvino_xml_path} (XML: {xml_size} bytes, BIN: {bin_size} bytes)")
        
        # Remove arquivo ONNX temporário para economizar espaço (opcional)
        try:
            os.remove(onnx_model_path)
            logging.debug(f"Arquivo ONNX temporário removido: {onnx_model_path}")
        except Exception as cleanup_error:
            logging.warning(f"Erro ao remover arquivo ONNX temporário: {cleanup_error}")
        
        return openvino_xml_path
        
    except subprocess.TimeoutExpired:
        logging.error(f"Timeout na conversão do modelo {model_name}")
        return None
    except Exception as e:
        logging.error(f"Erro na conversão Paddle->OpenVINO para {model_name}: {e}")
        return None