from multiprocessing import Process

import os
import re
import cv2
import logging

import uuid

from modules.detector import load_yolo
from modules.ocr import init_ocr

from app_ultils.logger import setup_logger

from app_ultils.config import load_config

from modules.postprocess import post_process_plate, choose_best_ocr_prediction

from modules.capture import init_capture, get_frame

from modules.validate import validate_bounding_box, validate_text

from modules.PlateObject import PlateObject

from dotenv import load_dotenv

from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

load_dotenv(ROOT_DIR / ".env")

CONFIG_FILE = ROOT_DIR / os.getenv("CONFIG_PATH", "config.json")
config = load_config(CONFIG_FILE)

LOGS_SAVE_DIR = ROOT_DIR / os.getenv("LOGS_SAVE_DIR", "log/")
os.makedirs(LOGS_SAVE_DIR, exist_ok=True)
setup_logger(LOGS_SAVE_DIR)

DB_CONNECTION = ROOT_DIR / os.getenv("DB_CONNECTION", "db/captures.db")
os.makedirs(DB_CONNECTION.parent, exist_ok=True)

PLATE_MODEL = ROOT_DIR / os.getenv("PLATE_MODEL")
OCR_RECOGNITION_MODEL = ROOT_DIR / os.getenv("OCR_RECOGNITION_MODEL")
OCR_DETECTION_MODEL = ROOT_DIR / os.getenv("OCR_DETECTION_MODEL")
OCR_CLASSIFICATION_MODEL = ROOT_DIR / os.getenv("OCR_CLASSIFICATION_MODEL")

CAPTURES_SAVE_DIR = ROOT_DIR / os.getenv("CAPTURES_SAVE_DIR")

USE_OCR_ANGLE_CLS = os.getenv("USE_OCR_ANGLE_CLS").lower() in ("true", "1", "yes")
USE_OCR_OPENVINO = os.getenv("USE_OCR_OPENVINO").lower() in ("true", "1", "yes")
USE_OCR_DETECTION = os.getenv("USE_OCR_DETECTION").lower() in ("true", "1", "yes")

SAVE_SUSPECT_DETECTIONS = os.getenv("SAVE_SUSPECT_DETECTIONS").lower() in ("true", "1", "yes")
SUSPECT_DETECTIONS_SAVE_DIR = ROOT_DIR / os.getenv("SUSPECT_DETECTIONS_SAVE_DIR")

INPUT_SOURCES = config["input_sources"]

def main(instance, input_name, input_endpoint):
    model = load_yolo(PLATE_MODEL)
    ocr = init_ocr(
        det_model_dir=str(OCR_DETECTION_MODEL),
        rec_model_dir=str(OCR_RECOGNITION_MODEL),
        cls_model_dir=str(OCR_CLASSIFICATION_MODEL),
        use_angle_cls=USE_OCR_ANGLE_CLS,
        use_openvino=USE_OCR_OPENVINO, 
        use_det=USE_OCR_DETECTION
    )

    cap, source_type = init_capture(input_endpoint)

    logging.info("Iniciando captura de vídeo")

    PlateObject.setup(
        db_connection=DB_CONNECTION, 
        captures_save_path=CAPTURES_SAVE_DIR, 
        suspect_detections_save_path=SUSPECT_DETECTIONS_SAVE_DIR
    )

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

        if results[0].boxes:
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
                            "type": 1, 
                            'input_name': input_name
                         })

                plate_crop = frame[y1:y2, x1:x2]

                adjusted = post_process_plate(plate_crop)

                prediction = ocr.ocr(adjusted, cls=USE_OCR_ANGLE_CLS, det=USE_OCR_DETECTION)

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
                                "type": 2, 
                                'input_name': input_name
                            })
                    
                    if PlateObject.instances.get(instance) is None:
                        PlateObject.instances[instance] = PlateObject(instance)

                    PlateObject.instances[instance].addCapture(
                        str(re.sub(r'[^a-zA-Z0-9]', '', plate_text)).upper(),
                        {'input_frame': frame, 'plate_bounding_box': [x1, y1, x2, y2], 'input_name': input_name}
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
    if len(list(INPUT_SOURCES.items())) > 0:
        processes = []

        for input_name, data in list(INPUT_SOURCES.items()):
            instance = data["instance"]
            input_endpoint = data["input_endpoint"]
            
            logging.info(f"Iniciando processo para {instance} com fonte {input_endpoint}")

            p = Process(target=main, args=(instance, input_name, input_endpoint))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        input_name, data = INPUT_SOURCES.items()[0]
        
        instance = data["instance"]
        input_endpoint = data["input_endpoint"]

        logging.info(f"Iniciando processo para {instance} com fonte {input_endpoint}")

        main(instance, input_name, input_endpoint)
