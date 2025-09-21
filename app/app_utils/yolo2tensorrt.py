import logging
import shutil
from pathlib import Path
from ultralytics import YOLO

# Ponto 3: Usando import relativo para robustez, buscando o config.py no diretório pai (app/)
from app_utils.config import APP_DIR

def yolo2tensorrt(yolo_model_path, output_base_dir=None):
    """
    Converte modelo YOLO para TensorRT com tratamento robusto de erros.
    
    Args:
        yolo_model_path (Path): Caminho para o modelo YOLO .pt
        output_base_dir (Path, optional): Diretório base para salvar modelos convertidos
        
    Returns:
        Path or None: Caminho para o arquivo .engine ou None se falhou
    """
    if not yolo_model_path or not yolo_model_path.exists():
        logging.warning(f"Modelo YOLO não existe: {yolo_model_path}")
        return None
        
    if output_base_dir is None:
        output_base_dir = APP_DIR / 'models' / 'plate' / 'engine'
    
    try:
        model_stem = yolo_model_path.stem
        engine_path = output_base_dir / (model_stem + '.engine')
        
        # Cria diretório se não existir
        output_base_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Convertendo YOLO para TensorRT: {model_stem}")
        
        # Remove arquivo existente se houver
        if engine_path.exists():
            try:
                engine_path.unlink()
                logging.debug(f"Arquivo engine existente removido: {engine_path}")
            except Exception as cleanup_error:
                logging.warning(f"Erro ao remover engine existente: {cleanup_error}")
        
        # Carrega e exporta o modelo
        try:
            model = YOLO(yolo_model_path)
            
            # Exporta para TensorRT com half=True para FP16
            exported_file = model.export(format='engine', half=True, device=0)
            
            # Converte para Path se necessário
            if isinstance(exported_file, str):
                exported_file = Path(exported_file)
            
            # Verifica se a exportação foi bem-sucedida
            if not exported_file.exists():
                logging.error(f"Arquivo engine não foi criado: {exported_file}")
                return None
            
            # Move para o local definitivo se necessário
            if exported_file != engine_path:
                shutil.move(str(exported_file), str(engine_path))
            
            # Verifica se o arquivo foi criado corretamente
            if not engine_path.exists():
                logging.error(f"Arquivo engine não foi movido: {engine_path}")
                return None
                
            # Verifica tamanho do arquivo
            engine_size = engine_path.stat().st_size
            
            if engine_size == 0:
                logging.error(f"Arquivo engine está vazio: {engine_path}")
                return None
            
            logging.info(f"Conversão YOLO->TensorRT concluída: {engine_path}")
            logging.info(f"Arquivo gerado - Engine: {engine_size} bytes")
            
            return engine_path
            
        except Exception as export_error:
            logging.error(f"Erro na exportação YOLO->TensorRT: {export_error}")
            return None
            
    except Exception as e:
        logging.error(f"Erro na conversão YOLO->TensorRT para {yolo_model_path.name}: {e}")
        return None