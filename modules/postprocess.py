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