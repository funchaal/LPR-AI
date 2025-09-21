import logging
import shutil
from pathlib import Path
from ultralytics import YOLO

# Ponto 3: Usando import relativo para robustez, buscando o config.py no diretório pai (app/)
from app_utils.config import settings, APP_DIR

def yolo2openvino(yolo_model_path, output_base_dir=None):
    """
    Converte modelo YOLO para OpenVINO com tratamento robusto de erros.
    
    Args:
        yolo_model_path (Path): Caminho para o modelo YOLO .pt
        output_base_dir (Path, optional): Diretório base para salvar modelos convertidos
        
    Returns:
        Path or None: Caminho para o diretório do modelo OpenVINO ou None se falhou
    """
    if not yolo_model_path or not yolo_model_path.exists():
        logging.warning(f"Modelo YOLO não existe: {yolo_model_path}")
        return None
        
    if output_base_dir is None:
        output_base_dir = APP_DIR / 'models' / 'plate' / 'openvino'
    
    try:
        model_stem = yolo_model_path.stem
        
        # Usa o nome padrão que a Ultralytics gera
        ov_model_dir_name = model_stem + '_openvino_model'
        ov_target_dir = output_base_dir / ov_model_dir_name
        ov_xml_path = ov_target_dir / (model_stem + '.xml')
        ov_bin_path = ov_target_dir / (model_stem + '.bin')
        
        # Cria diretório pai se não existir
        output_base_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Convertendo YOLO para OpenVINO: {model_stem}")
        
        # Remove diretório existente se houver
        if ov_target_dir.exists():
            try:
                shutil.rmtree(ov_target_dir)
                logging.debug(f"Diretório existente removido: {ov_target_dir}")
            except Exception as cleanup_error:
                logging.warning(f"Erro ao limpar diretório existente: {cleanup_error}")
        
        # Carrega e exporta o modelo
        try:
            model = YOLO(yolo_model_path)
            
            # Exporta para OpenVINO com half=True para FP16
            temp_exported_dir = model.export(format='openvino', half=True)
            
            # Converte para Path se necessário
            if isinstance(temp_exported_dir, str):
                temp_exported_dir = Path(temp_exported_dir)
            
            # Verifica se a exportação foi bem-sucedida
            if not temp_exported_dir.exists():
                logging.error(f"Diretório temporário não foi criado: {temp_exported_dir}")
                return None
            
            # Move para o local definitivo
            shutil.move(str(temp_exported_dir), str(ov_target_dir))
            
            # Verifica se os arquivos essenciais foram criados
            if not ov_xml_path.exists():
                logging.error(f"Arquivo .xml não foi gerado: {ov_xml_path}")
                return None
                
            if not ov_bin_path.exists():
                logging.warning(f"Arquivo .bin não encontrado: {ov_bin_path}")
            
            # Verifica tamanhos dos arquivos
            xml_size = ov_xml_path.stat().st_size
            bin_size = ov_bin_path.stat().st_size if ov_bin_path.exists() else 0
            
            if xml_size == 0:
                logging.error(f"Arquivo .xml está vazio: {ov_xml_path}")
                return None
            
            logging.info(f"Conversão YOLO->OpenVINO concluída: {ov_target_dir}")
            logging.info(f"Arquivos gerados - XML: {xml_size} bytes, BIN: {bin_size} bytes")
            
            return ov_target_dir
            
        except Exception as export_error:
            logging.error(f"Erro na exportação YOLO->OpenVINO: {export_error}")
            return None
            
    except Exception as e:
        logging.error(f"Erro na conversão YOLO->OpenVINO para {yolo_model_path.name}: {e}")
        return None