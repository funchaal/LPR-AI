from multiprocessing import Process

import json

import os
import re
import cv2
import logging

import uuid

from modules.detector import load_yolo

from app_utils.logger import setup_logger

from app_utils.config import load_config

from modules.postprocess import post_process_plate, choose_best_ocr_prediction, crop_margin

from modules.capture import init_capture, get_frame

from modules.validate import validate_bounding_box, validate_text

from modules.Tracking import Tracking

from dotenv import load_dotenv

from app_utils.env_utils import get_env_path, get_env_bool, get_env_int

from pathlib import Path

from modules.db_manager import CapturesDatabase

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

load_dotenv(ROOT_DIR / ".env")

CONFIG_FILE = ROOT_DIR / os.getenv("CONFIG_FILE", "config.json")
config = load_config(CONFIG_FILE)

LOGS_SAVE_DIR = ROOT_DIR / os.getenv("LOGS_SAVE_DIR", "log/")
os.makedirs(LOGS_SAVE_DIR, exist_ok=True)
setup_logger(LOGS_SAVE_DIR)

DB_CONNECTION = ROOT_DIR / os.getenv("DB_CONNECTION", "db/captures.db")
os.makedirs(DB_CONNECTION.parent, exist_ok=True)

PLATE_MODEL = get_env_path(ROOT_DIR, "PLATE_MODEL")

OCR_RECOGNITION_MODEL_DIR = get_env_path(ROOT_DIR, "OCR_RECOGNITION_MODELS_DIR")

OCR_CHAR_DICT_FILE = get_env_path(ROOT_DIR, "OCR_CHAR_DICT_FILE")
CHAR_CORRECTIONS_FILE = get_env_path(ROOT_DIR, "CHAR_CORRECTIONS_FILE")

with open(CHAR_CORRECTIONS_FILE, 'r', encoding='utf-8') as f:
    char_corrections = json.load(f)

READING_FORMATS = os.getenv("READING_FORMATS")
READING_FORMATS = READING_FORMATS.split(",") if READING_FORMATS else []

OCR_DETECTION_MODEL_DIR = get_env_path(ROOT_DIR, "OCR_DETECTION_MODELS_DIR")
OCR_CLASSIFICATION_MODEL_DIR = get_env_path(ROOT_DIR, "OCR_CLASSIFICATION_MODELS_DIR", "")

USE_OCR_OPENVINO = get_env_bool("USE_OCR_OPENVINO")
USE_OCR_DETECTION = get_env_bool("USE_OCR_DETECTION")
CROP_MARGIN = get_env_int("CROP_MARGIN")

OCR_DETECTION_MODEL = None
OCR_RECOGNITION_MODEL = None
OCR_CLASSIFICATION_MODEL = None

if USE_OCR_OPENVINO:
    if USE_OCR_DETECTION:
        OCR_DETECTION_MODEL = OCR_DETECTION_MODEL_DIR / 'openvino/det/en_PP-OCRv3_det_infer.xml'
    OCR_RECOGNITION_MODEL = OCR_RECOGNITION_MODEL_DIR / 'openvino/rec/en_PP-OCRv4_rec_infer.xml'
else:
    if USE_OCR_DETECTION:
        OCR_DETECTION_MODEL = OCR_DETECTION_MODEL_DIR / "paddlepaddle/det/en/en_PP-OCRv3_det_infer/"
    OCR_RECOGNITION_MODEL = OCR_RECOGNITION_MODEL_DIR / "paddlepaddle/rec/en/en_PP-OCRv4_rec_infer/"

USE_OCR_ANGLE_CLS = get_env_bool("USE_OCR_ANGLE_CLS")
if USE_OCR_ANGLE_CLS:
    OCR_CLASSIFICATION_MODEL = OCR_CLASSIFICATION_MODEL_DIR / "paddlepaddle/cls/ch_ppocr_mobile_v2.0_cls_infer/"

CAPTURES_SAVE_DIR = get_env_path(ROOT_DIR, "CAPTURES_SAVE_DIR")
SAVE_SUSPECT_DETECTIONS = get_env_bool("SAVE_SUSPECT_DETECTIONS")
SUSPECT_DETECTIONS_SAVE_DIR = get_env_path(ROOT_DIR, "SUSPECT_DETECTIONS_SAVE_DIR")

USE_CONTINUOUS_TRIES = get_env_bool("USE_CONTINUOUS_TRIES", False)

INPUT_SOURCES = config["input_sources"]

def process_source(instance_id, input_name, input_endpoint):
    if USE_OCR_OPENVINO:
        from modules.openvino_ocr import init_ocr
    else:
        from modules.ocr import init_ocr

    setup_logger(LOGS_SAVE_DIR)

    model = load_yolo(PLATE_MODEL)

    ocr = init_ocr(
        det_model_dir=str(OCR_DETECTION_MODEL),
        rec_model_dir=str(OCR_RECOGNITION_MODEL),
        cls_model_dir=str(OCR_CLASSIFICATION_MODEL),
        use_angle_cls=USE_OCR_ANGLE_CLS,
        use_det=USE_OCR_DETECTION, 
        char_dict_file=OCR_CHAR_DICT_FILE
    )

    cap, source_type = init_capture(input_endpoint)

    logging.info("Iniciando captura de vídeo")

    db_manager = CapturesDatabase(db_path=DB_CONNECTION)

    Tracking.setup(
        db_manager=db_manager,
        instance_id=instance_id,
        captures_save_path=CAPTURES_SAVE_DIR,
        suspect_detections_save_path=SUSPECT_DETECTIONS_SAVE_DIR,
        reading_formats=READING_FORMATS, 
        char_corrections=char_corrections
    )

    track = None

    while True:
        frame = get_frame(source_type, cap, input_endpoint)

        if frame is None:
            if source_type in ("stream", "camera"):
                logging.warning("Não foi possível ler frame da stream, tentando novamente...")
                continue
            else:
                logging.warning("Mensagem de erro ao obter frame do vídeo.")
                break

        Tracking.newFrame()

        results = model.predict(frame, verbose=False)

        frame_id = None

        if results[0].boxes:
            objects = results[0].boxes.data.tolist()

            for x1, y1, x2, y2, prob, cls in objects:
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                if SAVE_SUSPECT_DETECTIONS and not validate_bounding_box(x1, y1, x2, y2):
                    frame_id = str(uuid.uuid4())
                    Tracking.suspect_detections.append({ 
                        "frame_id": frame_id, 
                        "frame": frame, 
                        "coords": [x1, y1, x2, y2], 
                        "type": 1, 
                        'input_name': input_name
                        })

                plate_crop = frame[y1:y2, x1:x2]

                adjusted = post_process_plate(plate_crop)

                plate_text, score = None, None

                if USE_OCR_DETECTION:
                    prediction = ocr.ocr(adjusted, det=USE_OCR_DETECTION, cls=USE_OCR_ANGLE_CLS)

                    if prediction and prediction[0]:
                        plate_text, score = choose_best_ocr_prediction(prediction[0])
                    else:
                        continue
                else:
                    prediction = ocr.ocr(crop_margin(adjusted, margin_percent=CROP_MARGIN), det=USE_OCR_DETECTION, cls=USE_OCR_ANGLE_CLS)

                    if prediction and prediction[0]:
                        first_item = prediction[0][0]  # exemplo: ('TLLEL', 0.32)
                        if isinstance(first_item, tuple) and len(first_item) >= 2:
                            plate_text, score = first_item
                    else:
                        continue

                if SAVE_SUSPECT_DETECTIONS and not validate_text(plate_text):
                    if not frame_id:
                        frame_id = str(uuid.uuid4())
                    Tracking.suspect_detections.append({ 
                        "frame_id": frame_id, 
                        "frame": frame, 
                        "coords": [x1, y1, x2, y2], 
                        "type": 2, 
                        'input_name': input_name
                    })

                if not Tracking.trackings:
                    track = Tracking()
                    Tracking.trackings[track.id] = track

                Tracking.trackings[track.id].addCapture(
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

def main():
    """
    Esta é a função "gerente". Ela apenas cria e gerencia os processos.
    """
    if len(list(INPUT_SOURCES.items())) > 1:
        processes = []

        for input_name, data in list(INPUT_SOURCES.items()):
            instance = data["instance"]
            input_endpoint = data["input_endpoint"]
            
            logging.info(f"Iniciando processo para {instance} com fonte {input_endpoint}")

            # --- MUDANÇA PRINCIPAL: O target agora é a função trabalhadora ---
            p = Process(target=process_source, args=(instance, input_name, input_endpoint))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        print('hello')
        input_name, data = list(INPUT_SOURCES.items())[0]
        instance_id = data["instance"]
        input_endpoint = data["input_endpoint"]

        logging.info(f"Iniciando processo para {instance_id} com fonte {input_endpoint}")
        # Chama a função trabalhadora diretamente
        process_source(instance_id, input_name, input_endpoint)

if __name__ == '__main__':
    main()
