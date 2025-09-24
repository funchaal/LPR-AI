# modules/ocr.py
from paddleocr import TextRecognition, TextDetection
import logging
import os
from app_utils.optimize_ocr_model import optimize_ocr_model
from app_utils.config import settings
from app_utils.logger import setup_logger


def init_ocr(det_model_dir, rec_model_dir, use_det=True, device='cpu', char_dict_file=None, **kwargs):
    """
    Inicializa o PaddleOCR 3.0 usando as novas classes TextDetection e TextRecognition.
    Este script é chamado pelo main.py somente quando OCR_BACKEND='paddle'.
    
    Args:
        det_model_dir (str): Caminho para o modelo de detecção.
        rec_model_dir (str): Caminho para o modelo de reconhecimento.
        cls_model_dir (str, optional): Caminho para o modelo de classificação de ângulo.
        use_angle_cls (bool, optional): Se deve usar a classificação de ângulo.
        use_det (bool, optional): Se deve usar o modelo de detecção.
        char_dict_file (str, optional): Caminho para o arquivo de dicionário de caracteres.
        device (str, optional): O dispositivo de computação ('cpu', 'gpu:0', 'gpu:1', etc.). Default 'cpu'.
        **kwargs: Aceita argumentos extras para compatibilidade.
    
    Returns:
        OCRWrapper: Wrapper que fornece interface compatível com o código existente.
    """
    
    # Otimiza modelos baseado no dispositivo
    setup_logger()
    if device == 'cpu':
        try:
            # Tenta otimizar os modelos para OpenVINO
            optimized_det_model = optimize_ocr_model(det_model_dir, device) if det_model_dir else None
            optimized_rec_model = optimize_ocr_model(rec_model_dir, device) if rec_model_dir else None
            
            # Se conseguiu otimizar para OpenVINO, usa o init_openvino_ocr
            if ((optimized_det_model and optimized_det_model.endswith('.xml')) or not det_model_dir) and \
               (optimized_rec_model and optimized_rec_model.endswith('.xml')):
                
                from modules.openvino_ocr import init_openvino_ocr
                logging.info("Usando OpenVINO OCR otimizado para CPU")
                
                return init_openvino_ocr(
                    det_model_dir=optimized_det_model,
                    rec_model_dir=optimized_rec_model,
                    use_det=use_det,
                    char_dict_file=char_dict_file
                )
                
        except Exception as e:
            logging.error(f"Erro ao inicializar OCR otimizado: {e}")
            logging.info("Seguindo com PaddleOCR padrão")
    
    logging.info(f"Inicializando PaddleOCR 3.2.0 com device: {device}")
    
    # Inicializa o detector de texto se necessário
    text_detector = None
    use_tensorrt = (device == 'gpu') if settings.USE_OCR_TENSORRT else False
    use_hpi = settings.USE_OCR_HPI
    if use_det:
        try:
            if not det_model_dir:
                logging.info("Nenhum modelo de detecção personalizado fornecido. Usando modelo padrão.")
                text_detector = TextDetection(model_name='PP-OCRv5_mobile_det', device=device, enable_hpi=False, use_tensorrt=False)
            else:
                det_model_name = os.path.basename(det_model_dir)
                text_detector = TextDetection(model_name=det_model_name, device=device, model_dir=det_model_dir, enable_hpi=use_hpi, use_tensorrt=use_tensorrt)
                logging.info(f"TextDetection inicializado com modelo customizado: {det_model_name}")
        except Exception as e:
            logging.error(f"Erro ao inicializar TextDetection: {e}")
            raise

    # Inicializa o reconhecedor de texto
    try:
        if not rec_model_dir:
            logging.info("Nenhum modelo de reconhecimento personalizado fornecido. Usando modelo padrão.")
            text_recognizer = TextRecognition(model_name='en_PP-OCRv5_mobile_rec', device=device, enable_hpi=False, use_tensorrt=False)
        else:
            rec_model_name = os.path.basename(rec_model_dir)
            text_recognizer = TextRecognition(model_name=rec_model_name, device=device, model_dir=rec_model_dir, enable_hpi=use_hpi, use_tensorrt=use_tensorrt)
            logging.info(f"TextRecognition inicializado com modelo customizado: {rec_model_name}")
    except Exception as e:
        logging.error(f"Erro ao inicializar TextRecognition: {e}")
        raise
    
    # Retorna wrapper que mantém compatibilidade com a interface antiga
    return OCRWrapper(
        text_detector=text_detector,
        text_recognizer=text_recognizer,
        use_det=use_det
    )

class OCRWrapper:
    """
    Wrapper que mantém compatibilidade com a interface do PaddleOCR antigo
    enquanto usa internamente as novas classes TextDetection e TextRecognition.
    """
    
    def __init__(self, text_detector, text_recognizer, use_det=True):
        self.text_detector = text_detector
        self.text_recognizer = text_recognizer
        self.use_det = use_det
    
    def ocr(self, img):
        """
        Método principal que mantém compatibilidade com a interface antiga do PaddleOCR.
        
        Args:
            img: Imagem para processar
            det (bool): Se deve usar detecção de texto
            cls (bool): Se deve usar classificação de ângulo (não implementado na nova versão)
            
        Returns:
            Lista no formato compatível com PaddleOCR antigo
        """
        
        if self.use_det and self.text_detector:
            # Modo com detecção: primeiro detecta, depois reconhece cada região
            try:
                # Detecta regiões de texto
                detection_results = self.text_detector.predict(img)
                
                if not detection_results or len(detection_results) == 0:
                    logging.debug("Nenhuma região de texto detectada")
                    return [[]]
                
                # Lista para armazenar resultados no formato antigo
                ocr_results = []
                
                for detection_result in detection_results:
                    # Extrai as coordenadas da detecção
                    try:
                        # O formato pode variar, vamos tentar diferentes formatos
                        if detection_result.get('bbox') is not None:
                            # Usa a bounding box para fazer crop da imagem
                            bbox = detection_result.bbox
                            
                            # Converte coordenadas para inteiros
                            x1, y1, x2, y2 = map(int, [
                                min(bbox[0], bbox[2]), min(bbox[1], bbox[3]),
                                max(bbox[0], bbox[2]), max(bbox[1], bbox[3])
                            ])
                            
                            # Faz crop da região detectada
                            if x2 > x1 and y2 > y1:
                                text_region = img[y1:y2, x1:x2]
                                
                                # Reconhece o texto na região
                                recognition_results = self.text_recognizer.predict(text_region)
                                
                                if recognition_results and len(recognition_results) > 0:
                                    recognition_result = recognition_results[0]
                                    
                                    # Extrai texto e confiança
                                    text = recognition_result.get('rec_text', '')
                                    confidence = recognition_result.get('rec_score', 0.0)
                                    
                                    # Formato compatível: [coordenadas, (texto, confiança)]
                                    formatted_result = [
                                        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],  # coordenadas
                                        (text, confidence)  # texto e confiança
                                    ]
                                    ocr_results.append(formatted_result)
                        
                    except Exception as e:
                        logging.warning(f"Erro ao processar região detectada: {e}")
                        continue
                
                return [ocr_results] if ocr_results else [[]]
                
            except Exception as e:
                logging.error(f"Erro durante detecção: {e}")
                return [[]]
        
        else:
            # Modo sem detecção: reconhece diretamente a imagem inteira
            try:
                recognition_results = self.text_recognizer.predict(img)
                
                if not recognition_results or len(recognition_results) == 0:
                    logging.debug("Nenhum texto reconhecido")
                    return [[]]
                
                # Pega o primeiro resultado
                recognition_result = recognition_results[0]

                # Extrai texto e confiança
                text = recognition_result.get('rec_text', '')
                confidence = recognition_result.get('rec_score', 0.0)
                
                # Formato compatível para modo sem detecção
                return [[(text, confidence)]]
                
            except Exception as e:
                logging.error(f"Erro durante reconhecimento: {e}")
                return [[]]