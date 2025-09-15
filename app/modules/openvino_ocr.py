import os
import cv2
import numpy as np
from openvino.runtime import Core, PartialShape

from app_utils.config import settings

class OCR:
    """
    Classe final que encapsula a lógica de inferência OCR com OpenVINO.
    Implementa pós-processamento de detecção manual sem dependência do PaddleOCR.
    Usa pré-processamento manual para detecção e reconhecimento para máxima estabilidade.
    """
    def __init__(self, det_model_dir, rec_model_dir, cls_model_dir=None, use_angle_cls=False, use_det=True, char_dict_path=None):
        self.ie = Core()
        self.use_det = use_det
        self.use_angle_cls = use_angle_cls
        
        # Parâmetros para pós-processamento de detecção
        self.thresh = 0.3
        self.box_thresh = 0.6
        self.max_candidates = 1000
        self.unclip_ratio = 3.0

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

        if char_dict_path:
            self.character = self._load_char_dict(char_dict_path)
        else:
            self.character = ['blank'] + [str(i) for i in range(10)] + list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['unk']

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
        img_tensor = (img_tensor / 255.0 - 0.485) / 0.229  # Normalização padrão
        return np.expand_dims(img_tensor, axis=0), scale

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

    def _box_score_fast(self, bitmap, _box):
        """
        Calcula a pontuação de confiança para uma caixa delimitadora.
        """
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(int), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def _unclip(self, box, unclip_ratio):
        """
        Expande a caixa delimitadora usando o algoritmo Clipper.
        """
        import pyclipper
        poly = pyclipper.Pyclipper()
        poly.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        distance = cv2.contourArea(box) * unclip_ratio / cv2.arcLength(box, True)
        expanded = poly.Execute(pyclipper.CT_UNION, pyclipper.PFT_POSITIVE, pyclipper.PFT_POSITIVE, distance)
        return expanded

    def _get_mini_boxes(self, contour):
        """
        Obtém a caixa delimitadora mínima para um contorno.
        """
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def _postprocess_detection(self, pred, shape_list):
        """
        Pós-processamento manual para detecção de texto (substitui DBPostProcess).
        """
        segmentation = pred > self.thresh
        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            mask = segmentation[batch_index]
            boxes, scores = self._boxes_from_bitmap(pred[batch_index], mask, src_w, src_h)
            boxes_batch.append({'points': boxes})
        return boxes_batch

    def _boxes_from_bitmap(self, pred, bitmap, dest_width, dest_height):
        """
        Extrai caixas delimitadoras de um bitmap de segmentação.
        """
        bitmap = bitmap.astype(np.uint8)
        height, width = bitmap.shape
        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        num_contours = min(len(contours), self.max_candidates)
        boxes = []
        scores = []

        for index in range(num_contours):
            contour = contours[index]
            points, sside = self._get_mini_boxes(contour)
            if sside < 5:  # Muito pequeno
                continue
            
            points = np.array(points)
            
            # Calcular score
            score = self._box_score_fast(pred, points.reshape(-1, 2))
            if score < self.box_thresh:
                continue
                
            # Expandir a caixa
            box = self._unclip(points.reshape(-1, 2).astype(int), self.unclip_ratio)
            if not box:
                continue
            box = box[0]
            
            # Obter caixa mínima novamente após expansão  
            box, sside = self._get_mini_boxes(np.array(box).reshape(-1, 1, 2))
            if sside < 5 + 2:
                continue
                
            box = np.array(box)
            
            # Converter coordenadas para escala original
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            
            boxes.append(box)
            scores.append(score)
            
        return boxes, scores

    def ocr(self, frame):
        if self.use_det:
            original_h, original_w = frame.shape[:2]
            # Usa nosso pré-processamento manual
            det_input, scale = self._preprocess_det_manual(frame)

            det_output_tensor = self.det_compiled_model.output(0)
            det_preds = self.det_compiled_model(det_input)[det_output_tensor]

            # Pós-processamento manual (substitui DBPostProcess)
            resized_h, resized_w = det_input.shape[2:]
            ratio_h = resized_h / original_h
            ratio_w = resized_w / original_w
            shape_info = [[original_h, original_w, ratio_h, ratio_w]]
            post_result = self._postprocess_detection(det_preds, shape_info)
            
            dt_boxes = post_result[0]['points']
            if not dt_boxes: return [[]]

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

def init_openvino_ocr(det_model_dir=None, rec_model_dir=None, use_det=True, char_dict_file=None):
    """
    Função de fábrica para inicializar e retornar um objeto OCR.
    """

    def to_openvino_path(model_dir: str) -> str | None:
        if not model_dir:
            return None
        model_name = os.path.basename(model_dir)
        return os.path.join(model_dir.replace("paddlepaddle", "openvino"), model_name + ".xml")

    openvino_det_model_dir = to_openvino_path(det_model_dir)
    openvino_rec_model_dir = to_openvino_path(rec_model_dir)

    ocr_engine = OCR(
        det_model_dir=str(openvino_det_model_dir),
        rec_model_dir=str(openvino_rec_model_dir),
        use_det=use_det,
        char_dict_path=char_dict_file
    )

    return ocr_engine