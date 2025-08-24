import cv2
import numpy as np
from openvino.runtime import Core, PartialShape

from paddleocr.ppocr.postprocess import DBPostProcess

class OCR:
    """
    Classe final que encapsula a lógica de inferência OCR com OpenVINO.
    Usa o helper DBPostProcess do 'paddleocr' para máxima precisão na detecção.
    Usa pré-processamento manual para detecção e reconhecimento para máxima estabilidade.
    """
    def __init__(self, det_model_dir, rec_model_dir, cls_model_dir, use_angle_cls, use_det, char_dict_path):
        self.ie = Core()
        self.use_det = use_det
        self.use_angle_cls = use_angle_cls

        if self.use_det:
            print("Carregando modelo de Detecção (OpenVINO)...")
            det_model = self.ie.read_model(model=det_model_dir)
            self.det_compiled_model = self.ie.compile_model(model=det_model, device_name="CPU")

        if self.use_angle_cls:
            # Lógica de carregamento do CLS aqui...
            pass
            
        print("Carregando e configurando modelo de Reconhecimento (OpenVINO)...")
        rec_model = self.ie.read_model(model=rec_model_dir)
        new_shape = PartialShape([-1, 3, 48, -1])
        rec_model.reshape({rec_model.inputs[0]: new_shape})
        self.rec_compiled_model = self.ie.compile_model(model=rec_model, device_name="CPU")

        self.character = self._load_char_dict(char_dict_path)
        if self.use_det:
            self.postprocess_op = DBPostProcess(thresh=0.3, box_thresh=0.6, max_candidates=1000, unclip_ratio=3)

    def _get_rotate_crop_image(self, img, points):
        points = np.float32(points)
        width = int(np.round(np.linalg.norm(points[0] - points[1])))
        height = int(np.round(np.linalg.norm(points[1] - points[2])))
        if height > width:
            width, height = height, width
            src_pts = points.copy()
            points[0], points[1], points[2], points[3] = src_pts[1], src_pts[2], src_pts[3], src_pts[0]
        dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(points, dst_pts)
        return cv2.warpPerspective(img, M, (width, height))

    def _load_char_dict(self, char_dict_path):
        with open(char_dict_path, "rb") as f:
            char_list = [line.decode('utf-8').strip() for line in f.readlines()]
        return ['blank'] + char_list + ['unk']
    
    def _preprocess_det_manual(self, img):
        """Pré-processamento manual e explícito para o modelo de detecção."""
        limit_side_len = 960
        h, w, _ = img.shape
        scale = limit_side_len / max(h, w)
        img_resized = cv2.resize(img, (int(w * scale), int(h * scale)))
        h_resized, w_resized, _ = img_resized.shape
        pad_h = (32 - h_resized % 32) % 32
        pad_w = (32 - w_resized % 32) % 32
        img_padded = np.pad(img_resized, ((0, pad_h), (0, pad_w), (0, 0)), 'constant', constant_values=0)
        img_tensor = img_padded.transpose((2, 0, 1)).astype(np.float32)
        return np.expand_dims(img_tensor, axis=0)

    def _preprocess_rec_manual(self, img_crop_list):
        """Pré-processamento manual e estável para o modelo de reconhecimento."""
        img_h = 48
        widths = [int(img.shape[1] * (img_h / img.shape[0])) for img in img_crop_list]
        max_width = max(widths) if widths else 0
        if max_width == 0: return None # Evita erro se a lista de recortes estiver vazia

        batch_imgs = []
        for i, img in enumerate(img_crop_list):
            resized_img = cv2.resize(img, (widths[i], img_h))
            norm_img = (resized_img.astype(np.float32) / 255.0 - 0.5) / 0.5
            padded_img = np.pad(norm_img, ((0, 0), (0, max_width - widths[i]), (0, 0)), 'constant', constant_values=0)
            batch_imgs.append(padded_img)
        
        batch_tensor = np.stack(batch_imgs, axis=0)
        return batch_tensor.transpose((0, 3, 1, 2))

    def _postprocess_recognition(self, rec_preds):
        texts, scores = [], []
        for pred in rec_preds:
            text, score, count = "", 0.0, 0
            preds_idx = np.argmax(pred, axis=1)
            preds_prob = np.max(pred, axis=1)
            last_char_idx = 0
            for i in range(len(preds_idx)):
                char_idx = preds_idx[i]
                if char_idx > 0 and char_idx != last_char_idx:
                    text += self.character[char_idx]
                    score += preds_prob[i]
                    count += 1
                last_char_idx = char_idx
            texts.append(text)
            scores.append(score / count if count > 0 else 0.0)
        return texts, scores

    def ocr(self, frame , det=None, cls=None):
        if self.use_det:
            original_h, original_w = frame.shape[:2]
            # Usa nosso pré-processamento manual
            det_input = self._preprocess_det_manual(frame)

            det_output_tensor = self.det_compiled_model.output(0)
            det_preds = self.det_compiled_model(det_input)[det_output_tensor]

            # O pós-processamento continua usando o helper do paddleocr para máxima precisão
            resized_h, resized_w = det_input.shape[2:]
            ratio_h = resized_h / original_h
            ratio_w = resized_w / original_w
            shape_info = [[original_h, original_w, ratio_h, ratio_w]]
            post_result = self.postprocess_op({'maps': det_preds}, shape_list=shape_info)
            
            dt_boxes = post_result[0]['points']
            if not dt_boxes.any(): return [[]]

            img_crop_list = [self._get_rotate_crop_image(frame, box) for box in dt_boxes]

            if self.use_angle_cls and img_crop_list:
                # Lógica para classificação de ângulo...
                pass

            if not img_crop_list: return [[]]
            
            rec_batch = self._preprocess_rec_manual(img_crop_list)
            if rec_batch is None: return [[]]
            
            rec_output_tensor = self.rec_compiled_model.output(0)
            rec_preds = self.rec_compiled_model(rec_batch)[rec_output_tensor]
            texts, scores = self._postprocess_recognition(rec_preds)
            
            results = []
            for box, text, score in zip(dt_boxes, texts, scores):
                formatted_box = [[int(p[0]), int(p[1])] for p in box]
                results.append([formatted_box, (text, float(score))])
            return [results]
        else:
            rec_batch = self._preprocess_rec_manual([frame])
            if rec_batch is None: return [[('', 0.0)]]
            rec_output_tensor = self.rec_compiled_model.output(0)
            rec_preds = self.rec_compiled_model(rec_batch)[rec_output_tensor]
            texts, scores = self._postprocess_recognition(rec_preds)
            if not texts: return [[('', 0.0)]]
            return [[(texts[0], float(scores[0]))]]

def init_ocr(det_model_dir=None, rec_model_dir=None, cls_model_dir=None, use_angle_cls=False, use_det=True, char_dict_file=None):
    """
    Função de fábrica para inicializar e retornar um objeto OCR.
    """
    
    if not rec_model_dir:
        raise ValueError("O caminho para o modelo de reconhecimento (rec_model_dir) é obrigatório.")
    
    if use_det and not det_model_dir:
        raise ValueError("O uso da detecção (use_det=True) requer o caminho para o modelo de detecção (det_model_dir).")

    ocr_engine = OCR(
        det_model_dir=det_model_dir,
        rec_model_dir=rec_model_dir,
        cls_model_dir=cls_model_dir,
        use_angle_cls=use_angle_cls,
        use_det=use_det,
        char_dict_path=char_dict_file
    )
    return ocr_engine

# --- Exemplo de Uso ---
if __name__ == '__main__':
    # --- CONFIGURE SEU TESTE AQUI ---
    USE_OCR_DETECTION = True # Mude para True ou False para testar os dois modos
    
    if USE_OCR_DETECTION:
        image_path = 'test_image.jpg' 
    else:
        image_path = 'test_image.jpg'
    # --- FIM DA CONFIGURAÇÃO ---

    OCR_DETECTION_MODEL = 'models_openvino/en_PP-OCRv3_det_infer.xml'
    OCR_RECOGNITION_MODEL = 'models_openvino/en_PP-OCRv4_rec_infer.xml'
    OCR_CLASSIFICATION_MODEL = None 
    USE_OCR_ANGLE_CLS = False
    
    try:
        print(f"Inicializando o motor OCR (use_det={USE_OCR_DETECTION})...")
        ocr = init_ocr(
            det_model_dir=str(OCR_DETECTION_MODEL),
            rec_model_dir=str(OCR_RECOGNITION_MODEL),
            cls_model_dir=str(OCR_CLASSIFICATION_MODEL),
            use_angle_cls=USE_OCR_ANGLE_CLS,
            use_det=USE_OCR_DETECTION
        )
        print("Motor OCR pronto.")

        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Imagem não encontrada em {image_path}")

        print("\nExecutando OCR na imagem...")
        result = ocr.ocr(frame)
        print("OCR concluído.")

        print("\nResultado:")
        print(result)

        image_with_results = frame.copy()
        
        if result and result[0]:
            if USE_OCR_DETECTION:
                print("Modo Detecção: Desenhando caixas delimitadoras...")
                for line in result[0]:
                    box = np.array(line[0]).astype(np.int32)
                    text = f"{line[1][0]} ({line[1][1]:.2f})"
                    cv2.polylines(image_with_results, [box], isClosed=True, color=(0, 255, 0), thickness=2)
                    cv2.putText(image_with_results, text, (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                print("Modo Apenas Reconhecimento: Escrevendo texto na imagem...")
                text, score = result[0][0]
                label = f"{text} ({score:.2f})"
                
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                img_h, img_w = image_with_results.shape[:2]
                cv2.putText(image_with_results, label, ((img_w - w) // 2, (img_h + h) // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imwrite("resultado_inferencia_final.jpg", image_with_results)
        print("\nImagem com resultados salva em 'resultado_inferencia_final.jpg'")

    except FileNotFoundError as e:
        print(f"\nERRO: {e}")
    except Exception as e:
        import traceback
        print(f"\nOcorreu um erro inesperado: {e}")
        traceback.print_exc()