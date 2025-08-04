from multiprocessing import Process

import os
import re
import cv2
import logging
import numpy as np

import uuid

from modules.detector import load_yolo
from modules.ocr import init_ocr

from app_ultils.logger import setup_logger

from app_ultils.config import load_config

from modules.postprocess import post_process_plate, choose_best_ocr_prediction

from modules.capture import init_capture, get_frame

from modules.validate import validate_bounding_box, validate_text

from modules.PlateObject import PlateObject

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CONFIG_PATH = 'config/config.json'
config = load_config(CONFIG_PATH)

LOG_PATH = config["logging_path"]
setup_logger(LOG_PATH)

PLATE_MODEL_PATH = config["plate_detection_model"]
INPUT_SOURCES = config["input_sources"]
CAPTURES_SAVE_PATH = config["captures_save_path"]
DB_CONNECTION = config["db_connection"]

OCR_RECOGNITION_MODEL = config["ocr_recognition_model"]
OCR_DETECTION_MODEL = config["ocr_detection_model"]
OCR_CLASSIFICATION_MODEL = config["ocr_classification_model"]
USE_ANGLE_CLS = config["use_angle_cls"]
USE_OPENVINO = config["use_openvino"]
USE_DETECTION = config["use_detection"]

SAVE_SUSPECT_DETECTIONS = config["save_suspect_detections"]
SUSPECT_DETECTIONS_SAVE_PATH = config.get("suspect_detections_save_path")

def main(instance, input_name, input_endpoint):
    model = load_yolo(PLATE_MODEL_PATH)
    ocr = init_ocr(
        det_model_dir=OCR_DETECTION_MODEL,
        rec_model_dir=OCR_RECOGNITION_MODEL,
        cls_model_dir=OCR_CLASSIFICATION_MODEL,
        use_angle_cls=USE_ANGLE_CLS,
        use_openvino=USE_OPENVINO, 
        use_det=USE_DETECTION
    )

    cap, source_type = init_capture(input_endpoint)

    logging.info("Iniciando captura de vídeo")

    PlateObject.setup(db_connection=DB_CONNECTION, captures_save_path=CAPTURES_SAVE_PATH, suspect_detections_save_path=SUSPECT_DETECTIONS_SAVE_PATH)

    while True:
        frame = get_frame(source_type, cap, input_endpoint)

        if frame is None:
            if source_type in ("stream", "camera"):
                logging.warning("Não foi possível ler frame da stream, tentando novamente...")
                continue
            else:
                logging.warning("Mensagem de erro ao obter frame do vídeo.")
                break

        PlateObject.newFrame(instance)

        results = model.predict(frame, verbose=False)

        frame_id = None

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            objects = results[0].boxes.data.tolist()

            for x1, y1, x2, y2, prob, cls in objects:
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                if not validate_bounding_box(x1, y1, x2, y2):
                    if SAVE_SUSPECT_DETECTIONS:
                        frame_id = str(uuid.uuid4())
                        PlateObject.suspect_detections.append({ 
                            "frame_id": frame_id, 
                            "frame": frame, 
                            "coords": [x1, y1, x2, y2], 
                            "type": 1
                         })
                    continue

                plate_crop = frame[y1:y2, x1:x2]

                adjusted = post_process_plate(plate_crop)

                prediction = ocr.ocr(adjusted, cls=USE_ANGLE_CLS, det=USE_DETECTION)

                if prediction and prediction[0]:
                    plate_text, score = choose_best_ocr_prediction(prediction[0])

                    if not validate_text(plate_text):
                        if SAVE_SUSPECT_DETECTIONS:
                            if not frame_id:
                                frame_id = str(uuid.uuid4())
                            PlateObject.suspect_detections.append({ 
                                "frame_id": frame_id, 
                                "frame": frame, 
                                "coords": [x1, y1, x2, y2], 
                                "type": 2
                            })
                        continue
                    
                    if PlateObject.instances.get(instance) is None:
                        PlateObject.instances[instance] = PlateObject(instance)

                    PlateObject.instances[instance].addCapture(
                        str(re.sub(r'[^a-zA-Z0-9]', '', plate_text)).upper(),
                        {'inputFrame': frame, 'plateBoundingBox': [x1, y1, x2, y2]}
                    )
                else:
                    logging.debug(f"Nenhuma placa reconhecida.")

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            logging.info("Tecla 'q' pressionada, encerrando captura")
            break
    
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    logging.info("Processamento finalizado")

if __name__ == '__main__':
    processes = []

    for input_name, data in INPUT_SOURCES.items():
        instance = data["instance"]
        input_endpoint = data["input_endpoint"]
        
        logging.info(f"Iniciando processo para {instance} com fonte {input_endpoint}")

        p = Process(target=main, args=(instance, input_name, input_endpoint))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
