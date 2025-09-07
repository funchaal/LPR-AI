# engine.py
import os
import cv2
import logging
import numpy as np
import re
import uuid
import time
from pathlib import Path

# Módulos da aplicação
from modules.detector import load_yolo
from modules.postprocess import post_process_plate, choose_best_ocr_prediction, crop_margin
from modules.preprocess import draw_polygonal_mask
from modules.capture import init_capture, get_frame
from modules.validate import validate_bounding_box, validate_text
from modules.Tracking import Tracking
from modules.db_manager import CapturesDatabase
from app_utils.logger import setup_logger

# Importa a instância única de configurações do arquivo config.py
from app_utils.config import settings

# Garante a compatibilidade com certas bibliotecas de deep learning em alguns ambientes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def process_source(instance_id: str, input_name: str, input_endpoint: str, model_path: Path, polygons: list = None):
    """
    Processa uma única fonte de vídeo, desde a captura até a análise e salvamento.
    Esta função contém o loop principal de processamento de frames.
    """
    # Configura o logger para este processo específico, garantindo logs separados se necessário
    setup_logger(settings.LOGS_SAVE_DIR)
    logging.info(f"[{input_name}] Iniciando processo para instância '{instance_id}'")
    logging.info(f"[{input_name}] Backend de OCR selecionado: {settings.OCR_BACKEND.upper()}")
    logging.info(f"[{input_name}] Dispositivo de Computação selecionado: {settings.COMPUTE_DEVICE.upper()}")
    
    # Carrega modelo YOLO (detecção de placas)
    model = load_yolo(model_path)

    # --- Lógica de Inicialização Dinâmica do OCR ---
    if settings.OCR_BACKEND == 'openvino':
        from modules.openvino_ocr import init_ocr as ocr_initializer
        logging.info(f"[{input_name}] Carregando inicializador de OCR do módulo 'openvino_ocr'.")
    else: # Padrão é 'paddle'
        from modules.ocr import init_ocr as ocr_initializer
        logging.info(f"[{input_name}] Carregando inicializador de OCR do módulo 'ocr' (Paddle).")
    
    common_args = {
        'det_model_dir': str(settings.OCR_DETECTION_MODEL) if settings.OCR_DETECTION_MODEL else None,
        'rec_model_dir': str(settings.OCR_RECOGNITION_MODEL) if settings.OCR_RECOGNITION_MODEL else None,
        'cls_model_dir': str(settings.OCR_CLASSIFICATION_MODEL) if settings.OCR_CLASSIFICATION_MODEL else None,
        'use_angle_cls': settings.USE_OCR_ANGLE_CLS,
        'use_det': settings.USE_OCR_DETECTION,
        'char_dict_file': settings.OCR_CHAR_DICT_FILE
    }

    if settings.OCR_BACKEND == 'paddle':
        common_args.update({'backend': settings.OCR_BACKEND, 'device': settings.COMPUTE_DEVICE})

    ocr = ocr_initializer(**common_args)
    
    setup_logger(settings.LOGS_SAVE_DIR)
    # Inicializa captura de vídeo e gerenciador de banco de dados
    cap, source_type = init_capture(input_endpoint)
    db_manager = CapturesDatabase(db_path=settings.DB_CONNECTION)

    # Configura o módulo de Tracking com os parâmetros carregados
    Tracking.setup(
        db_manager=db_manager,
        instance_id=instance_id,
        captures_save_path=settings.CAPTURES_SAVE_DIR,
        suspect_detections_save_path=settings.SUSPECT_DETECTIONS_SAVE_DIR,
        reading_formats=settings.READING_FORMATS,
        char_corrections=settings.CHAR_CORRECTIONS,
        use_continuous_tries=settings.USE_CONTINUOUS_TRIES,
        api_endpoint=settings.API_ENDPOINT
    )

    track = None
    logging.info(f"[{input_name}] Iniciando loop de captura de vídeo para a fonte: {input_endpoint}")

    if settings.SHOW_CAPTURES and settings.CALCULATE_FPS:
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0.0

    # --- Loop Principal de Processamento ---
    while True:
        frame = get_frame(source_type, cap, input_endpoint)

        if frame is None:
            if source_type in ("stream", "camera"):
                logging.warning(f"[{input_name}] Não foi possível ler frame da stream, tentando reconectar...")
                cap, source_type = init_capture(input_endpoint) # Tenta reconectar
                continue
            else:
                logging.info(f"[{input_name}] Fim do vídeo ou erro ao obter frame. Encerrando.")
                break

        processed_frame = draw_polygonal_mask(frame, polygons)
        Tracking.newFrame()
        
        # Executa a predição do YOLO no dispositivo correto
        results = model.predict(processed_frame, device=settings.COMPUTE_DEVICE, verbose=False)
        frame_id = None

        if results and results[0].boxes:
            objects = results[0].boxes.data.tolist()
            for x1, y1, x2, y2, prob, cls in objects:
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                if settings.SAVE_SUSPECT_DETECTIONS and not validate_bounding_box(x1, y1, x2, y2):
                    frame_id = str(uuid.uuid4())
                    Tracking.suspect_detections.append({
                        "frame_id": frame_id, "frame": frame, "coords": [x1, y1, x2, y2],
                        "type": 1, 'input_name': input_name
                    })

                plate_crop = frame[y1:y2, x1:x2]
                adjusted = post_process_plate(plate_crop)
                plate_text, score = None, None

                if settings.USE_OCR_DETECTION:
                    prediction = ocr.ocr(adjusted, det=True, cls=settings.USE_OCR_ANGLE_CLS)
                    if prediction and prediction[0]:
                        plate_text, score = choose_best_ocr_prediction(prediction[0])
                    else:
                        continue
                else:
                    prediction = ocr.ocr(crop_margin(adjusted, margin_percent=settings.CROP_MARGIN), det=False, cls=settings.USE_OCR_ANGLE_CLS)
                    if prediction and prediction[0]:
                        first_item = prediction[0][0]
                        if isinstance(first_item, tuple) and len(first_item) >= 2:
                            plate_text, score = first_item
                        else:
                            continue
                
                if settings.SAVE_SUSPECT_DETECTIONS and not validate_text(plate_text):
                    if not frame_id:
                        frame_id = str(uuid.uuid4())
                    Tracking.suspect_detections.append({
                        "frame_id": frame_id, "frame": frame, "coords": [x1, y1, x2, y2],
                        "type": 2, 'input_name': input_name
                    })

                if not Tracking.trackings:
                    track = Tracking()
                    Tracking.trackings[track.id] = track
                elif not track or track.id not in Tracking.trackings:
                    track = Tracking()
                    Tracking.trackings[track.id] = track

                Tracking.trackings[track.id].addCapture(
                    str(re.sub(r'[^a-zA-Z0-9]', '', plate_text)).upper(),
                    {'input_frame': frame, 'plate_bounding_box': [x1, y1, x2, y2], 'input_name': input_name, 'reading': plate_text}
                )
        else:
            logging.debug("Nenhuma placa detectada neste frame.")

        if settings.SHOW_CAPTURES:

            # --- LÓGICA DE CÁLCULO DE FPS ---
            if settings.CALCULATE_FPS:
                fps_frame_count += 1
                # Calcula o FPS a cada segundo para uma leitura mais estável
                if (time.time() - fps_start_time) > 1.0:
                    fps = fps_frame_count / (time.time() - fps_start_time)
                    # logging.info(f"[{input_name}] FPS: {fps:.2f}")
                    # Reseta o contador e o tempo
                    fps_start_time = time.time()
                    fps_frame_count = 0

            display_frame = frame.copy()
            if polygons:
                polygon_pts = [np.array(p, dtype=np.int32) for p in polygons]
                cv2.polylines(display_frame, polygon_pts, isClosed=True, color=(0, 255, 0), thickness=2)

            # --- DESENHA O FPS NA TELA ---
            if settings.CALCULATE_FPS:
                fps_text = f"FPS: {fps:.2f}"
                cv2.putText(
                    display_frame, 
                    fps_text, 
                    (10, 30), # Posição (x, y) no canto superior esquerdo
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, # Tamanho da fonte
                    (0, 255, 0), # Cor (Verde)
                    2 # Espessura da linha
                )

            cv2.imshow(f"Frame - {input_name}", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logging.info(f"[{input_name}] Tecla 'q' pressionada, encerrando captura.")
                break
    
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    logging.info(f"[{input_name}] Processamento finalizado.")