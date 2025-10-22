import os
import cv2
import numpy as np
from openvino.runtime import Core, PartialShape
import logging

from app_utils.config import settings

class OCR:
    """
    Classe final que encapsula a lógica de inferência OCR com OpenVINO.
    Implementa pós-processamento de detecção manual sem dependência do PaddleOCR.
    Usa pré-processamento manual para detecção e reconhecimento para máxima estabilidade.
    """
    def __init__(self, det_model_dir, rec_model_dir, logger: logging.Logger, cls_model_dir=None, use_angle_cls=False, use_det=True, char_dict_path=None):
        self.logger = logger
        self.ie = Core()
        self.use_det = use_det
        self.use_angle_cls = use_angle_cls
        
        # Parâmetros para pós-processamento de detecção
        self.thresh = 0.3
        self.box_thresh = 0.6
        self.max_candidates = 1000
        self.unclip_ratio = 3.0

        if self.use_det:
            self.logger.info("Carregando modelo de Detecção (OpenVINO)...")
            det_model = self.ie.read_model(model=det_model_dir)
            self.det_compiled_model = self.ie.compile_model(model=det_model, device_name="CPU")

        if self.use_angle_cls:
            pass
            
        self.logger.info("Carregando e configurando modelo de Reconhecimento (OpenVINO)...")
        rec_model = self.ie.read_model(model=rec_model_dir)
        new_shape = PartialShape([-1, 3, 48, -1])
        rec_model.reshape({rec_model.inputs[0]: new_shape})
        self.rec_compiled_model = self.ie.compile_model(model=rec_model, device_name="CPU")

        if char_dict_path:
            self.character = self._load_char_dict(char_dict_path)
        else:
            self.character = ['blank'] + [str(i) for i in range(10)] + list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['unk']

        self.logger.debug(f"Caminho do dicionário recebido: {char_dict_path}")
        self.logger.debug(f"Tamanho do dicionário de caracteres carregado: {len(self.character)}")

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
    
    def _preprocess_detection(self, img):
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

    def _preprocess_recognition(self, img_crop_list):
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
            
                # Condição para pular caracteres repetidos ou o 'blank' token
                if char_idx > 0 and char_idx != last_char_idx:
                
                    # Pega o caractere correspondente ao índice
                    char = self.character[char_idx]
                
                    # Adiciona uma nova condição para ignorar o caractere 'unk'
                    if char != 'unk':
                        text += char
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
        Expande a caixa delimitadora usando expansão geométrica simples.
        Substitui pyclipper para evitar crashes.
        """
        if len(box) < 4:
            return None
        
        try:
            # Calcula o centro da caixa
            box = np.array(box, dtype=np.float32)
            center = np.mean(box, axis=0)
            
            # Calcula a distância de expansão baseada na área e perímetro
            area = cv2.contourArea(box)
            perimeter = cv2.arcLength(box, True)
            
            if perimeter == 0:
                return None
            
            # Fator de expansão
            distance = area * unclip_ratio / perimeter
            
            # Expande cada ponto afastando-o do centro
            expanded_box = []
            for point in box:
                # Vetor do centro para o ponto
                vector = point - center
                # Normaliza e multiplica pela distância
                if np.linalg.norm(vector) > 0:
                    vector_norm = vector / np.linalg.norm(vector)
                    new_point = point + vector_norm * distance
                    expanded_box.append(new_point.tolist())
                else:
                    expanded_box.append(point.tolist())
            
            return [expanded_box]
        except Exception as e:
            self.logger.warning(f"Erro ao expandir caixa: {e}")
            return None

    def _get_mini_boxes(self, contour):
        """
        Obtém a caixa delimitadora mínima para um contorno.
        """
        # Garante que o contour seja um array numpy com tipo correto
        if not isinstance(contour, np.ndarray):
            contour = np.array(contour)
        
        # Garante que seja float32 ou int32 (requerido pelo OpenCV)
        if contour.dtype not in [np.float32, np.int32]:
            contour = contour.astype(np.float32)
        
        # Garante que tenha pelo menos 3 pontos
        if len(contour) < 3:
            return None, 0
        
        # Reshape se necessário para formato correto (N, 1, 2) ou (N, 2)
        if len(contour.shape) == 3:
            if contour.shape[1] != 1 or contour.shape[2] != 2:
                contour = contour.reshape(-1, 2)
        elif len(contour.shape) == 2:
            if contour.shape[1] != 2:
                return None, 0
        else:
            return None, 0
        
        try:
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
        except Exception as e:
            self.logger.warning(f"Erro em _get_mini_boxes: {e}")
            return None, 0

    def _postprocess_detection(self, pred, shape_list):
        """
        Pós-processamento manual para detecção de texto (substitui DBPostProcess).
        """
        # Remove dimensões extras se necessário (batch, 1, height, width) -> (batch, height, width)
        if len(pred.shape) == 4:
            pred = pred.squeeze(1)
        
        segmentation = pred > self.thresh
        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            mask = segmentation[batch_index]
            
            if len(mask.shape) > 2:
                mask = mask.squeeze()
            
            boxes, scores = self._boxes_from_bitmap(pred[batch_index], mask, src_w, src_h)
            boxes_batch.append({'points': boxes})
        return boxes_batch

    def _boxes_from_bitmap(self, pred, bitmap, dest_width, dest_height):
        """
        Extrai caixas delimitadoras de um bitmap de segmentação.
        """
        # CORREÇÃO: Garante que pred e bitmap sejam 2D
        if len(pred.shape) > 2:
            pred = pred.squeeze()
        if len(bitmap.shape) > 2:
            bitmap = bitmap.squeeze()
            
        bitmap = bitmap.astype(np.uint8)
        height, width = bitmap.shape
        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        num_contours = min(len(contours), self.max_candidates)
        boxes = []
        scores = []

        for index in range(num_contours):
            contour = contours[index]
            
            # Validação básica do contorno
            if len(contour) < 4:
                continue
                
            result = self._get_mini_boxes(contour)
            if result[0] is None:
                continue
            points, sside = result
            
            if sside < 5:  # Muito pequeno
                continue
            
            points = np.array(points)
            
            # Validação dos pontos
            if len(points) != 4:
                continue
            
            # Calcular score
            try:
                score = self._box_score_fast(pred, points.reshape(-1, 2))
            except Exception as e:
                self.logger.warning(f"Erro ao calcular score: {e}")
                continue
                
            if score < self.box_thresh:
                continue
                
            # Expandir a caixa
            try:
                box = self._unclip(points.reshape(-1, 2), self.unclip_ratio)
            except Exception as e:
                self.logger.warning(f"Erro ao expandir caixa: {e}")
                continue
                
            if not box or len(box) == 0:
                continue
            box = np.array(box[0], dtype=np.float32)
            
            # Validação após expansão
            if len(box) < 4:
                continue
            
            # Obter caixa mínima novamente após expansão
            # Garante formato correto antes de chamar _get_mini_boxes
            try:
                box_reshaped = box.reshape(-1, 2).astype(np.float32)
                result = self._get_mini_boxes(box_reshaped)
                if result[0] is None:
                    continue
                box, sside = result
            except Exception as e:
                self.logger.warning(f"Erro ao obter mini boxes após expansão: {e}")
                continue
                
            if sside < 5 + 2:
                continue
                
            box = np.array(box)
            
            # Validação final
            if len(box) != 4:
                continue
            
            # Converter coordenadas para escala original
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            
            boxes.append(box)
            scores.append(score)
            
        return boxes, scores

    def ocr(self, frame):
        if self.use_det:
            original_h, original_w = frame.shape[:2]
            det_input, scale = self._preprocess_detection(frame)

            det_output_tensor = self.det_compiled_model.output(0)
            det_preds = self.det_compiled_model(det_input)[det_output_tensor]

            resized_h, resized_w = det_input.shape[2:]
            ratio_h = resized_h / original_h
            ratio_w = resized_w / original_w
            shape_info = [[original_h, original_w, ratio_h, ratio_w]]
            post_result = self._postprocess_detection(det_preds, shape_info)
            
            dt_boxes = post_result[0]['points']
            if not dt_boxes: return [[]]

            img_crop_list = [self._get_rotate_crop_image(frame, box) for box in dt_boxes]

            if self.use_angle_cls and img_crop_list:
                pass

            if not img_crop_list: return [[]]
            
            rec_batch = self._preprocess_recognition(img_crop_list)
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
            rec_batch = self._preprocess_recognition([frame])
            if rec_batch is None: return [[('', 0.0)]]
            rec_output_tensor = self.rec_compiled_model.output(0)
            rec_preds = self.rec_compiled_model(rec_batch)[rec_output_tensor]
            texts, scores = self._postprocess_recognition(rec_preds)
            if not texts: return [[('', 0.0)]]
            return [[(texts[0], float(scores[0]))]]

def init_openvino_ocr(det_model_dir=None, rec_model_dir=None, use_det=True, char_dict_file=None, logger: logging.Logger = None, **kwargs):
    """
    Função de fábrica para inicializar um objeto OCR usando modelos OpenVINO.

    Esta versão foi ajustada para encontrar os modelos .xml dentro de uma estrutura
    de subdiretórios específica (ex: .../PP-OCRv5_rec/PP-OCRv5_rec_openvino_model/).
    """

    def find_openvino_model_path(base_dir: str) -> str | None:
        """
        Encontra o caminho para o arquivo .xml do modelo OpenVINO a partir de um diretório base.
        
        Args:
            base_dir (str): O diretório principal do modelo (ex: 'models/PP-OCRv5_rec').

        Returns:
            str | None: O caminho completo para o arquivo .xml ou None se não for encontrado.
        """
        if not base_dir or not os.path.isdir(base_dir):
            return None
        
        # 1. Constrói o nome esperado para o subdiretório do OpenVINO
        #    os.path.normpath remove barras finais (/) para que basename funcione corretamente
        base_name = os.path.basename(os.path.normpath(base_dir))
        openvino_subdir_name = f"{base_name}_openvino_model"
        openvino_subdir_path = os.path.join(base_dir, openvino_subdir_name)

        if not os.path.isdir(openvino_subdir_path):
            logger.warning(f"Subdiretório OpenVINO esperado não foi encontrado em: {openvino_subdir_path}")
            return None

        # 2. Procura pelo arquivo .xml dentro do subdiretório
        for filename in os.listdir(openvino_subdir_path):
            if filename.endswith(".xml"):
                model_path = os.path.join(openvino_subdir_path, filename)
                logger.info(f"Modelo OpenVINO encontrado: {model_path}")
                return model_path
        
        logger.warning(f"Nenhum arquivo .xml de modelo encontrado no diretório: {openvino_subdir_path}")
        return None

    # --- Lógica Principal ---

    # Encontra os caminhos dos modelos de detecção (se aplicável) e reconhecimento
    openvino_det_model_path = find_openvino_model_path(det_model_dir) if use_det else None
    openvino_rec_model_path = find_openvino_model_path(rec_model_dir)

    # Verifica se os modelos essenciais foram encontrados para evitar erros na inicialização
    if not openvino_rec_model_path:
        raise FileNotFoundError(f"Não foi possível encontrar o modelo de reconhecimento OpenVINO (.xml) no diretório base: {rec_model_dir}")
    
    if use_det and not openvino_det_model_path:
        raise FileNotFoundError(f"Uso de detecção está ativo, mas não foi possível encontrar o modelo de detecção OpenVINO (.xml) no diretório base: {det_model_dir}")

    # Inicializa a classe do motor de OCR com os caminhos corretos para os arquivos .xml
    ocr_engine = OCR(
        det_model_dir=openvino_det_model_path,
        rec_model_dir=openvino_rec_model_path,
        use_det=use_det,
        char_dict_path=char_dict_file,
        logger=logger
    )

    logger.info("Motor de OCR OpenVINO inicializado com sucesso.")
    return ocr_engine