import logging
import shutil
from pathlib import Path
from ultralytics import YOLO

from app_utils.config import APP_DIR

def yolo2onnx(yolo_model_path, yolo_inference_device, output_base_dir=None):
    """
    Converte modelo YOLO para ONNX com tratamento robusto de erros.
    
    Args:
        yolo_model_path (Path): Caminho para o modelo YOLO .pt
        output_base_dir (Path, optional): Diretório base para salvar modelos convertidos
        
    Returns:
        Path or None: Caminho para o arquivo .onnx ou None se falhou
    """
    if not yolo_model_path or not yolo_model_path.exists():
        logging.warning(f"Modelo YOLO não existe: {yolo_model_path}")
        return None
        
    if output_base_dir is None:
        output_base_dir = APP_DIR / 'models' / 'plate' / 'onnx'
    
    try:
        model_stem = yolo_model_path.stem
        onnx_path = output_base_dir / (model_stem + '.onnx')
        
        # Cria diretório se não existir
        output_base_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Convertendo YOLO para ONNX: {model_stem}")
        
        # Remove arquivo existente se houver
        if onnx_path.exists():
            try:
                onnx_path.unlink()
                logging.debug(f"Arquivo ONNX existente removido: {onnx_path}")
            except Exception as cleanup_error:
                logging.warning(f"Erro ao remover ONNX existente: {cleanup_error}")
        
        # Carrega e exporta o modelo
        try:
            model = YOLO(yolo_model_path)
            
            # Exporta para ONNX com half=True para FP16
            exported_file = model.export(format='onnx', half=True, device=yolo_inference_device)
            
            # Converte para Path se necessário
            if isinstance(exported_file, str):
                exported_file = Path(exported_file)
            
            # Verifica se a exportação foi bem-sucedida
            if not exported_file.exists():
                logging.error(f"Arquivo ONNX não foi criado: {exported_file}")
                return None
            
            # Move para o local definitivo se necessário
            if exported_file != onnx_path:
                shutil.move(str(exported_file), str(onnx_path))
            
            # Verifica se o arquivo foi criado corretamente
            if not onnx_path.exists():
                logging.error(f"Arquivo ONNX não foi movido: {onnx_path}")
                return None
                
            # Verifica tamanho do arquivo
            onnx_size = onnx_path.stat().st_size
            
            if onnx_size == 0:
                logging.error(f"Arquivo ONNX está vazio: {onnx_path}")
                return None
            
            logging.info(f"Conversão YOLO->ONNX concluída: {onnx_path}")
            logging.info(f"Arquivo gerado - ONNX: {onnx_size} bytes")
            
            return onnx_path
            
        except Exception as export_error:
            logging.error(f"Erro na exportação YOLO->ONNX: {export_error}")
            return None
            
    except Exception as e:
        logging.error(f"Erro na conversão YOLO->ONNX para {yolo_model_path.name}: {e}")
        return None
