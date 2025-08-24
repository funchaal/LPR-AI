import cv2
import numpy as np

ALPHA = 1
BETA = 20

def post_process_plate(plate_frame):
    adjusted = cv2.convertScaleAbs(plate_frame, alpha=ALPHA, beta=BETA)

    return adjusted

def choose_best_ocr_prediction(predictions):
    plate_text = ''
    score = 0
    max_area = 0

    for det in predictions:
        box_coords, (det_text, det_conf) = det
        area = cv2.contourArea(np.array(box_coords, dtype=np.float32))

        if area > max_area:
            max_area = area
            plate_text = det_text
            score = det_conf

    return plate_text, score

import cv2

def crop_margin(frame, margin_percent):
    """
    Corta uma margem percentual da imagem para dentro.

    Args:
        frame (numpy.ndarray): Imagem OpenCV (BGR).
        margin_percent (int): Percentual da margem a cortar (0-100).

    Returns:
        numpy.ndarray: Imagem cropped.
    """
    if not 0 <= margin_percent <= 100:
        raise ValueError("margin_percent deve estar entre 0 e 100")

    h, w = frame.shape[:2]

    # Calcula pixels a cortar
    margin_h = int(h * margin_percent / 100)
    margin_w = int(w * margin_percent / 100)

    # Crop
    cropped_frame = frame[margin_h:h - margin_h, margin_w:w - margin_w]

    return cropped_frame