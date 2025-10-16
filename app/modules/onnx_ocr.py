# modules/onnx_ocr.py
# Contém a implementação completa do backend de OCR usando ONNX Runtime.

import onnxruntime as ort
import numpy as np
import cv2
import logging
import os

# --- CLASSES DE INFERÊNCIA ONNX ---

class ONNXRecognizer:
    """Encapsula um modelo de reconhecimento de texto ONNX."""
    def __init__(self, model_path, providers, char_dict_file):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Arquivo do modelo de reconhecimento não encontrado: {model_path}")
        if not os.path.exists(char_dict_file):
            raise FileNotFoundError(f"Arquivo de dicionário não encontrado: {char_dict_file}")

        self.character_list = self._load_character_dict(char_dict_file)
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        logging.info(f"ONNXRecognizer usa o provider: {self.session.get_providers()[0]}")

    def _preprocess(self, img):
        expected_height = self.input_shape[2]
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w, _ = img.shape
        ratio = expected_height / h
        new_width = int(w * ratio)
        resized_img = cv2.resize(img, (new_width, expected_height))
        normalized_img = (resized_img.astype('float32') / 255.0 - 0.5) / 0.5
        transposed_img = normalized_img.transpose((2, 0, 1))
        return np.expand_dims(transposed_img, axis=0)

    def _load_character_dict(self, dict_path):
        with open(dict_path, 'r', encoding='utf-8') as f:
            chars = [line.strip() for line in f]
        chars.insert(0, '[blank]')
        return chars

    def _decode(self, model_output):
        pred_indices = np.argmax(model_output[0], axis=1)
        text = ''
        last_index = 0
        for index in pred_indices:
            if index != 0 and index != last_index:
                text += self.character_list[index]
            last_index = index
        probs = np.max(model_output[0], axis=1)
        confidence = float(np.mean(probs))
        return text, confidence

    def predict(self, img):
        preprocessed_img = self._preprocess(img)
        result = self.session.run([self.output_name], {self.input_name: preprocessed_img})
        text, confidence = self._decode(result[0])
        return [{'rec_text': text, 'rec_score': confidence}]


class ONNXDetector:
    """Encapsula um modelo de detecção de texto ONNX (baseado em DBNet)."""
    def __init__(self, model_path, providers):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Arquivo do modelo de detecção não encontrado: {model_path}")
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        logging.info(f"ONNXDetector usa o provider: {self.session.get_providers()[0]}")
        self.max_side_len = 960
        self.threshold = 0.3
        self.box_thresh = 0.6

    def _preprocess(self, img):
        src_h, src_w, _ = img.shape
        h, w = src_h, src_w
        if max(h, w) > self.max_side_len:
            ratio = self.max_side_len / max(h, w)
            h, w = int(h * ratio), int(w * ratio)
        h = max(int(round(h / 32) * 32), 32)
        w = max(int(round(w / 32) * 32), 32)
        resized_img = cv2.resize(img, (w, h))
        
        # Adicionamos dtype=np.float32 para garantir o tipo correto
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized_img = ((resized_img.astype('float32') / 255.0) - mean) / std
        
        transposed_img = normalized_img.transpose((2, 0, 1))
        return np.expand_dims(transposed_img, axis=0), (src_h, src_w)

    def _postprocess(self, model_output, original_shape):
        pred = model_output[0, 0, :, :]
        segmentation = pred > self.threshold
        contours, _ = cv2.findContours((segmentation).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        src_h, src_w = original_shape
        pred_h, pred_w = pred.shape
        scale_h, scale_w = src_h / pred_h, src_w / pred_w
        boxes = []
        for contour in contours:
            points = contour.reshape(-1, 2)
            box_points = (points * [scale_w, scale_h]).astype(np.int32)
            x, y, w, h = cv2.boundingRect(box_points)
            boxes.append([x, y, x + w, y + h])
        return boxes

    def predict(self, img):
        preprocessed_img, original_shape = self._preprocess(img)
        result = self.session.run([self.output_name], {self.input_name: preprocessed_img})
        boxes = self._postprocess(result[0], original_shape)
        return boxes


# --- WRAPPER E FUNÇÃO DE INICIALIZAÇÃO PÚBLICA ---

class OCRWrapper:
    """Wrapper que mantém a compatibilidade com a interface de predição anterior."""
    def __init__(self, text_detector, text_recognizer, use_det=True):
        self.text_detector = text_detector
        self.text_recognizer = text_recognizer
        self.use_det = use_det
    
    def ocr(self, img):
        if not self.use_det:
            try:
                rec_res = self.text_recognizer.predict(img)[0]
                return [[(rec_res['rec_text'], rec_res['rec_score'])]]
            except Exception as e:
                logging.error(f"Erro no reconhecimento ONNX (sem detecção): {e}", exc_info=True)
                return [[]]
        
        try:
            detection_boxes = self.text_detector.predict(img)
            if not detection_boxes: return [[]]
            
            ocr_results = []
            for box in detection_boxes:
                x1, y1, x2, y2 = box
                if y2 > y1 and x2 > x1:
                    text_region = img[y1:y2, x1:x2]
                    rec_res = self.text_recognizer.predict(text_region)[0]
                    formatted_result = [
                        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        (rec_res['rec_text'], rec_res['rec_score'])
                    ]
                    ocr_results.append(formatted_result)
            return [ocr_results] if ocr_results else [[]]
        except Exception as e:
            logging.error(f"Erro no pipeline de OCR ONNX com detecção: {e}", exc_info=True)
            return [[]]


def init_onnx_ocr(det_model_dir, rec_model_dir, use_det=True, device='cpu', char_dict_file=None, **kwargs):
    """
    Inicializa o motor de OCR usando ONNX Runtime, encontrando os modelos .onnx
    em uma estrutura de subdiretórios específica.
    """

    def _find_onnx_model_path(base_dir: str) -> str | None:
        """
        Encontra o caminho para o arquivo .onnx a partir de um diretório base.
        Procura em: .../base_dir/base_dir_onnx_model/modelo.onnx
        """
        if not base_dir or not os.path.isdir(base_dir):
            return None
        
        # 1. Constrói o nome do subdiretório ONNX esperado
        base_name = os.path.basename(os.path.normpath(base_dir))
        onnx_subdir_name = f"{base_name}_onnx_model"
        onnx_subdir_path = os.path.join(base_dir, onnx_subdir_name)

        if not os.path.isdir(onnx_subdir_path):
            logging.warning(f"Subdiretório ONNX esperado não foi encontrado em: {onnx_subdir_path}")
            return None

        # 2. Procura pelo arquivo .onnx dentro do subdiretório
        for filename in os.listdir(onnx_subdir_path):
            if filename.endswith(".onnx"):
                model_path = os.path.join(onnx_subdir_path, filename)
                logging.info(f"Modelo ONNX encontrado: {model_path}")
                return model_path
        
        logging.warning(f"Nenhum arquivo .onnx de modelo encontrado no diretório: {onnx_subdir_path}")
        return None

    # --- Lógica Principal ---

    # Define os providers de execução com base no dispositivo
    providers = ['CUDAExecutionProvider'] if 'gpu' in device else ['CPUExecutionProvider']
    
    # Encontra os caminhos dos modelos de detecção e reconhecimento
    text_detector = None
    if use_det:
        det_model_path = _find_onnx_model_path(det_model_dir)
        if not det_model_path: 
            raise RuntimeError(f"Modelo de detecção .onnx não pôde ser encontrado na estrutura de diretórios de '{det_model_dir}'")
        text_detector = ONNXDetector(model_path=det_model_path, providers=providers)

    rec_model_path = _find_onnx_model_path(rec_model_dir)
    if not rec_model_path: 
        raise RuntimeError(f"Modelo de reconhecimento .onnx não pôde ser encontrado na estrutura de diretórios de '{rec_model_dir}'")
    
    text_recognizer = ONNXRecognizer(
        model_path=rec_model_path, 
        providers=providers, 
        char_dict_file=char_dict_file
    )
    
    logging.info("Motor de OCR ONNX inicializado com sucesso.")
    return OCRWrapper(text_detector=text_detector, text_recognizer=text_recognizer, use_det=use_det)
