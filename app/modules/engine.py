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
from modules.capture import VideoSource
from modules.validate import validate_bounding_box, validate_text
from modules.Tracking import Tracking
from modules.db_manager import CapturesDatabase
from app_utils.logger import setup_logger
from modules.ocr import init_ocr

from app_utils.config import _validate_and_normalize_device  # Importa o objeto de configurações já validado
from app.app_utils.optimize_yolo_model import ensure_best_model

# Importa a instância única de configurações do arquivo config.py
from app_utils.config import settings

# Garante a compatibilidade com certas bibliotecas de deep learning em alguns ambientes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def process_source(instance_id: str, input_name: str, input_endpoint: str, input_user: str = None, input_password: str = None, polygons: list = None, yolo_device: str = 'cpu', ocr_device: str = 'cpu'):
    """
    Processa uma única fonte de vídeo, desde a captura até a análise e salvamento.
    Esta função contém o loop principal de processamento de frames.

    Args:
        instance_id (str): ID da instância de captura.
        input_name (str): Nome amigável da fonte de entrada.
        input_endpoint (str): URL/caminho da fonte de vídeo.
        model_path (Path): Caminho para o modelo YOLO otimizado.
        polygons (list, optional): Lista de polígonos para mascarar a área de detecção. Defaults to None.
        device (str, optional): Dispositivo de computação a ser usado ('cpu', 'gpu:0', etc.). Defaults to 'cpu'.
    """
    # Configura o logger para este processo específico

    setup_logger(settings.LOGS_SAVE_DIR)

    logging.info(f"[{input_name}] Iniciando processo para instância '{instance_id}'")

    yolo_validated_device = _validate_and_normalize_device(yolo_device)
    yolo_inference_device = '0' if yolo_validated_device == 'gpu' else yolo_validated_device.replace('gpu:', '')

    # Carrega modelo YOLO (detecção de placas)
    model = load_yolo(settings.BASE_PLATE_MODEL, yolo_validated_device)

    ocr_validated_device = _validate_and_normalize_device(ocr_device)

    ocr_common_args = {
        'det_model_dir': str(settings.OCR_DETECTION_MODEL) if settings.OCR_DETECTION_MODEL else None,
        'rec_model_dir': str(settings.OCR_RECOGNITION_MODEL) if settings.OCR_RECOGNITION_MODEL else None,
        'use_det': settings.USE_OCR_DETECTION,
        'device': ocr_validated_device, 
        'char_dict_file': settings.OCR_CHAR_DICT_FILE
    }

    ocr = init_ocr(**ocr_common_args)
    
    setup_logger(settings.LOGS_SAVE_DIR)

    # Inicializa captura de vídeo e gerenciador de banco de dados
    video_source = VideoSource(
        input_endpoint, 
        username=input_user,     # opcional, se existir
        password=input_password  # opcional, se existir
    )

    db_manager = CapturesDatabase(db_path=settings.DB_CONNECTION)
    
    # Configura o módulo de Tracking com os parâmetros carregados
    Tracking.setup(
        db_manager=db_manager,
        instance_id=instance_id,
        captures_save_path=settings.CAPTURES_SAVE_DIR,
        reading_formats=settings.READING_FORMATS, 
        char_corrections=settings.CHAR_CORRECTIONS,
        readings_filter_regex=settings.READINGS_FILTER_REGEX,
        api_endpoint=settings.API_ENDPOINT,
        api_user=settings.API_USER,
        api_password=settings.API_PASSWORD,
        use_continuous_tries=settings.USE_CONTINUOUS_TRIES,
        suspect_detections_save_path=settings.SUSPECT_DETECTIONS_SAVE_DIR,
    )
    
    # Variável para rastrear o tracking atual
    current_track = None
    logging.info(f"[{input_name}] Iniciando loop de captura de vídeo para a fonte: {input_endpoint}")
    
    if settings.SHOW_CAPTURES and settings.CALCULATE_FPS:
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0.0

    # --- Loop Principal de Processamento ---
    while True:
        frame = video_source.get_frame()
        if frame is None:
            if video_source.source_type in ("stream", "camera"):
                logging.warning(f"[{input_name}] Tentando reconectar...")
                video_source = VideoSource(
                    input_endpoint, 
                    username=settings.STREAM_USER, 
                    password=settings.STREAM_PASSWORD
                )
                continue
            else:
                logging.info(f"[{input_name}] Fim do vídeo ou erro ao obter frame. Encerrando.")
                break

        processed_frame = draw_polygonal_mask(frame, polygons)
        
        # Chama newFrame para gerenciar timeouts dos trackings
        Tracking.newFrame()

        results = model.predict(processed_frame, device=yolo_inference_device, verbose=False)
        frame_id = None
        
        if results and results[0].boxes:
            objects = results[0].boxes.data.tolist()
            
            for x1, y1, x2, y2, prob, cls in objects:
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                
                # Salva detecções suspeitas (bounding box inválido)
                if settings.SAVE_SUSPECT_DETECTIONS and not validate_bounding_box(x1, y1, x2, y2):
                    frame_id = str(uuid.uuid4())
                    Tracking.suspect_detections.append({
                        "frame_id": frame_id, 
                        "frame": frame, 
                        "coords": [x1, y1, x2, y2],
                        "type": 1, 
                        'input_name': input_name
                    })

                if current_track and Tracking.trackings.get(current_track.id) is not None and Tracking.trackings.get(current_track.id).closing and Tracking.trackings.get(current_track.id).api_calls > 0:
                    current_track = Tracking()
                    Tracking.trackings[current_track.id] = current_track

                if current_track and Tracking.trackings.get(current_track.id) is not None and Tracking.trackings.get(current_track.id).closing:
                    Tracking.trackings.get(current_track.id).noFrameCount = 0
                    continue

                # Extrai e processa a região da placa
                plate_crop = frame[y1:y2, x1:x2]
                adjusted = post_process_plate(plate_crop)
                plate_text, score = '', None
                prediction = None
                
                # OCR na placa
                if settings.USE_OCR_DETECTION:
                    cropped_image = crop_margin(adjusted, margin_percent=settings.CROP_MARGIN)
                    prediction = ocr.ocr(cropped_image)
                else:
                    prediction = ocr.ocr(adjusted)

                if prediction and prediction[0]:
                    if settings.USE_OCR_DETECTION:
                        plate_text, score = choose_best_ocr_prediction(prediction[0])
                    else:
                        plate_text, score = prediction[0][0]

                # Limpa o texto da placa
                plate_text = str(re.sub(r'[^a-zA-Z0-9]', '', plate_text)).upper()

                if not plate_text:
                    logging.debug("Predição OCR vazia.")
                    continue
               
                # Salva detecções suspeitas (texto inválido)
                if settings.SAVE_SUSPECT_DETECTIONS and not validate_text(plate_text):
                    if not frame_id:
                        frame_id = str(uuid.uuid4())
                    Tracking.suspect_detections.append({
                        "frame_id": frame_id, 
                        "frame": frame, 
                        "coords": [x1, y1, x2, y2],
                        "type": 2, 
                        'input_name': input_name
                    })

                if not current_track or not Tracking.trackings or Tracking.trackings.get(current_track.id) is None:
                    current_track = Tracking()
                    Tracking.trackings[current_track.id] = current_track

                capture_data = {
                    'input_frame': frame, 
                    'plate_bounding_box': [x1, y1, x2, y2], 
                    'input_name': input_name, 
                    'reading': plate_text
                }

                current_track.addCapture(capture_data)
        else:
            logging.debug("Nenhuma placa detectada neste frame.")

        # Interface de visualização (se habilitada)
        if settings.SHOW_CAPTURES:
            if settings.CALCULATE_FPS:
                fps_frame_count += 1
                if (time.time() - fps_start_time) > 1.0:
                    fps = fps_frame_count / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    fps_frame_count = 0

            display_frame = frame.copy()

            # Desenha polígonos de máscara
            if polygons:
                polygon_pts = [np.array(p, dtype=np.int32) for p in polygons]
                cv2.polylines(display_frame, polygon_pts, isClosed=True, color=(0, 255, 0), thickness=2)

            # Exibe FPS
            if settings.CALCULATE_FPS:
                fps_text = f"FPS: {fps:.2f}"
                cv2.putText(display_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Exibe informações de tracking
            if current_track and current_track.id in Tracking.trackings:
                info_y = 70
                cv2.putText(display_frame, f"Tracking ID: {current_track.id[:8]}", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display_frame, f"Leituras: {len(current_track.readings)}", 
                           (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                if current_track.possibleReadings:
                    cv2.putText(display_frame, f"Melhor: {current_track.possibleReadings[0]}", 
                               (10, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Exibe contadores de trackings
            active_count = len(Tracking.trackings)
            cv2.putText(display_frame, f"Ativos: {active_count}", 
                       (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow(f"Frame - {input_name}", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logging.info(f"[{input_name}] Tecla 'q' pressionada, encerrando captura.")
                break
    
    # Cleanup ao final do processamento
    logging.info(f"[{input_name}] Finalizando processamento. Aguardando fechamento de trackings pendentes...")
    
    # Força o fechamento de todos os trackings ativos
    for track_id in list(Tracking.trackings.keys()):
        track = Tracking.trackings.get(track_id)
        if track and not track.closing:
            logging.info(f"[{input_name}] Forçando fechamento do tracking {track_id}")
            track._start_async_close()
    
    # Aguarda um tempo para que os fechamentos assíncronos sejam concluídos
    max_wait_time = 30  # segundos
    wait_start = time.time()
    
    while Tracking.trackings and (time.time() - wait_start) < max_wait_time:
        active_count = len(Tracking.trackings)
        if active_count > 0:
            logging.info(f"[{input_name}] Aguardando: {active_count} ativos.")
            time.sleep(1)
        else:
            break
    
    # Libera recursos
    video_source.release()
    cv2.destroyAllWindows()
    logging.info(f"[{input_name}] Processamento finalizado.")