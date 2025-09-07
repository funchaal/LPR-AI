# modules/ocr.py

from paddleocr import PaddleOCR
import logging

def init_ocr(det_model_dir, rec_model_dir, cls_model_dir=None, use_angle_cls=False, use_det=True, char_dict_file=None, device='cpu', **kwargs):
    """
    Inicializa o PaddleOCR usando o backend PADRÃO (não OpenVINO).
    Este script é chamado pelo main.py somente quando OCR_BACKEND='paddle'.

    Ele configura o uso de CPU ou GPU com base no parâmetro 'device'.

    Args:
        det_model_dir (str): Caminho para o modelo de detecção.
        rec_model_dir (str): Caminho para o modelo de reconhecimento.
        cls_model_dir (str, optional): Caminho para o modelo de classificação de ângulo.
        use_angle_cls (bool, optional): Se deve usar a classificação de ângulo.
        use_det (bool, optional): Se deve usar o modelo de detecção.
        char_dict_file (str, optional): Caminho para o arquivo de dicionário de caracteres.
        device (str, optional): O dispositivo de computação ('cpu' ou 'cuda'). Default 'cpu'.
        **kwargs: Aceita argumentos extras para compatibilidade (como 'backend') que não serão usados.
    """
    use_gpu = False
    if device == 'cuda':
        use_gpu = True
        logging.info("Configurando PaddleOCR para usar o backend Paddle com GPU (CUDA).")
    else:
        logging.info("Configurando PaddleOCR para usar o backend Paddle com CPU.")
        
    try:
        ocr = PaddleOCR(
            use_angle_cls=use_angle_cls,
            use_det=use_det,
            use_rec=True,
            lang="en", 
            det=use_det, 
            det_model_dir=det_model_dir if use_det else None,
            rec_model_dir=rec_model_dir,
            cls_model_dir=cls_model_dir if use_angle_cls else None,
            
            # --- Configurações de Dispositivo ---
            use_gpu=use_gpu, 
            use_fp16=False
                    )
        logging.info("Instância do PaddleOCR (backend padrão) inicializada com sucesso.")
        return ocr
    except Exception as e:
        logging.error(f"Erro ao inicializar OCR do backend Paddle: {e}")
        if "Cannot load GPU library" in str(e) or "CUBLAS" in str(e):
            logging.error("ERRO CRÍTICO: O PaddlePaddle-GPU não conseguiu encontrar as bibliotecas CUDA.")
            logging.error("Verifique se o NVIDIA Driver, CUDA Toolkit e cuDNN estão instalados corretamente.")
        raise
