# Importa bibliotecas necessárias
import os  # Interação com o sistema operacional
import cv2  # OpenCV para processamento de imagem
import logging  # Para registrar informações e erros
import numpy as np  # NumPy para manipulação de arrays
import re  # Expressões regulares para limpeza de texto
import time  # Para medir o tempo e adicionar pausas

# Importa módulos customizados da aplicação
from modules.postprocess import post_process_plate, choose_best_ocr_prediction, crop_margin, is_detection_stationary
from modules.preprocess import draw_polygonal_mask
from modules.capture import VideoSource
from modules.Tracking import Tracking
from modules.db_manager import CapturesDatabase
from app_utils.logger import setup_logger

# Importa a instância única de configurações do arquivo config.py
from app_utils.config import settings

# Garante a compatibilidade com certas bibliotecas de deep learning em alguns ambientes,
# evitando erros de inicialização duplicada da biblioteca KMP.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def process_source(instance_id: str, input_name: str, input_endpoint: str, input_user: str, input_password: str, polygons: list, yolo, ocr, yolo_inference_device):
    """
    Processa uma única fonte de vídeo, desde a captura até a análise e salvamento.
    Esta função contém o loop principal de processamento de frames.

    Args:
        instance_id (str): ID da instância de captura.
        input_name (str): Nome amigável da fonte de entrada.
        input_endpoint (str): URL/caminho da fonte de vídeo.
        polygons (list): Lista de polígonos para mascarar a área de detecção.
        yolo: Objeto do modelo YOLO para detecção de placas.
        ocr: Objeto do modelo OCR para reconhecimento de caracteres.
        yolo_inference_device (str): Dispositivo para inferência do YOLO (ex: 'cpu', 'cuda:0').
    """
    # Configura o logger para este processo específico, garantindo que os logs sejam
    # direcionados e formatados corretamente.
    setup_logger()

    logging.info(f"[{input_name}] Iniciando processo para instância '{instance_id}'")

    # Inicializa a captura de vídeo com as credenciais e o endpoint fornecidos.
    video_source = VideoSource(
        input_endpoint, 
        username=input_user,     # opcional, se existir
        password=input_password  # opcional, se existir
    )

    # Se o tipo de fonte não for reconhecido, encerra a execução para evitar erros.
    if video_source.source_type is None:
        exit(1)

    # Inicializa o gerenciador do banco de dados para salvar as capturas.
    db_manager = CapturesDatabase(db_path=settings.DB_CONNECTION)
    
    # Configura o módulo de Tracking com os parâmetros carregados das configurações.
    # O Tracking é responsável por agrupar detecções da mesma placa ao longo do tempo.
    Tracking.setup(
        db_manager=db_manager,
        instance_id=instance_id,
        captures_save_path=settings.CAPTURES_SAVE_DIR,
        reading_formats=settings.READING_FORMATS,
        char_corrections=settings.CHAR_CORRECTIONS,
        readings_filter_regex=settings.READINGS_FILTER_REGEX,
        max_no_frame_count=settings.MAX_NO_FRAME_COUNT,
        api_endpoint=settings.API_ENDPOINT,
        api_user=settings.API_USER,
        api_password=settings.API_PASSWORD,
        close_api_endpoint=settings.CLOSE_API_ENDPOINT,
        use_continuous_tries=settings.USE_CONTINUOUS_TRIES,
        save_suspect_detections=settings.SAVE_SUSPECT_DETECTIONS,
        suspect_detections_save_path=settings.SUSPECT_DETECTIONS_SAVE_DIR,
    )
    
    # Variável para manter o controle do objeto de tracking atual.
    current_track = None
    logging.info(f"[{input_name}] Iniciando loop de captura de vídeo para a fonte: {input_endpoint}")
    
    # Inicializa variáveis para o cálculo de FPS, se habilitado.
    if settings.CALCULATE_FPS:
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0.0
        last_logged_fps = 0.0

    # Dicionário para rastrear a estabilidade da placa detectada.
    stationary_plate_tracker = {
        'last_bbox': None,
        'stability_count': 0
    }
    
    # Variáveis para armazenar o estado anterior e evitar logs repetidos.
    last_plate_count = -1
    last_stationary_count = -1
    last_ocr_text = ""

    # --- Loop Principal de Processamento ---
    while True:
        # Obtém um novo frame da fonte de vídeo.
        frame = video_source.get_frame()

        # Se o frame for nulo, trata a situação dependendo do tipo de fonte.
        if frame is None:
            if video_source.source_type in ("rtsp", "http", "camera"):
                # Para streams, tenta reconectar após uma pausa.
                logging.warning(f"[{input_name}] Frame nulo, aguardando 2s para tentar reconectar...")
                time.sleep(2)
                continue
            else: # Para vídeos, significa que o arquivo terminou.
                logging.info(f"[{input_name}] Fim do vídeo ou erro fatal. Encerrando.")
                break

        # Aplica a máscara polígonal no frame, se houver polígonos definidos.
        processed_frame = draw_polygonal_mask(frame, polygons)
        
        # Informa ao módulo de Tracking que um novo frame chegou para gerenciar timeouts.
        Tracking.newFrame()

        # Realiza a predição com o modelo YOLO para encontrar placas no frame.
        results = yolo.predict(processed_frame, device=yolo_inference_device, verbose=False)
        
        # Se houver resultados e caixas de detecção (bounding boxes).
        if results and results[0].boxes:
            objects = results[0].boxes.data.tolist()
            num_plates = len(objects)

            # Loga o número de placas detectadas se for diferente do frame anterior.
            if num_plates != last_plate_count:
                logging.info(f"{num_plates} placa(s) detectada(s) neste frame.")
                last_plate_count = num_plates
            
            # Itera sobre cada objeto (placa) detectado.
            for x1, y1, x2, y2, prob, cls in objects:
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                # Loga a contagem de leituras estacionárias se mudar.
                if stationary_plate_tracker["stability_count"] != last_stationary_count:
                    logging.info(f'Sequencia de leituras estacionárias: {stationary_plate_tracker["stability_count"]}')
                    last_stationary_count = stationary_plate_tracker["stability_count"]

                # Verifica se a detecção é estacionária para evitar processamento repetido.
                if is_detection_stationary(
                    tracker_state=stationary_plate_tracker,
                    new_bbox=(x1, y1, x2, y2),
                    max_diff=settings.STABILITY_MAX_COORDINATE_DIFFERENCE,
                    stationary_frame_threshold=settings.STATIONARY_FRAME_THRESHOLD 
                ):
                    # Reseta o contador de "sem frame" para o tracking atual, se existir.
                    if current_track and Tracking.trackings.get(current_track.id) is not None:
                        Tracking.trackings.get(current_track.id).noFrameCount = 0

                    logging.debug(f"[{input_name}] Placa estacionária detectada. Ignorando.")
                    continue

                # Lógica para iniciar um novo tracking se o anterior foi fechado após chamada de API.
                if current_track and Tracking.trackings.get(current_track.id) is not None and Tracking.trackings.get(current_track.id).closing and Tracking.trackings.get(current_track.id).api_calls > 0:
                    current_track = Tracking()
                    Tracking.trackings[current_track.id] = current_track

                # Se o tracking atual está fechando, reseta o contador e continua.
                if current_track and Tracking.trackings.get(current_track.id) is not None and Tracking.trackings.get(current_track.id).closing:
                    Tracking.trackings.get(current_track.id).noFrameCount = 0
                    logging.debug('NoFrameCount zerado.')
                    continue

                # Extrai e processa a região da placa (ROI - Region of Interest).
                plate_crop = frame[y1:y2, x1:x2]
                adjusted = post_process_plate(plate_crop)
                
                # Realiza o OCR na imagem da placa processada.
                if settings.USE_OCR_DETECTION:
                    cropped_image = crop_margin(adjusted, margin_percent=settings.CROP_MARGIN)
                    prediction = ocr.ocr(cropped_image)
                else:
                    prediction = ocr.ocr(adjusted)

                # Extrai o texto e a pontuação da predição do OCR.
                if prediction and prediction[0]:
                    if settings.USE_OCR_DETECTION:
                        plate_text, score = choose_best_ocr_prediction(prediction[0])
                    else:
                        plate_text, score = prediction[0][0]

                # Limpa o texto da placa, removendo caracteres não alfanuméricos.
                plate_text = str(re.sub(r'[^a-zA-Z0-9]', '', plate_text)).upper()

                # Se o OCR não retornar texto, ignora a detecção.
                if not plate_text:
                    if last_ocr_text:
                        logging.info("Predição OCR vazia.")
                        last_ocr_text = ''
                    continue

                # Loga o texto do OCR se for diferente do anterior.
                if plate_text and plate_text != last_ocr_text:
                    logging.info(f"OCR lido: {plate_text}")
                    last_ocr_text = plate_text

                # Garante que existe um objeto de tracking ativo.
                if not current_track or not Tracking.trackings or Tracking.trackings.get(current_track.id) is None:
                    current_track = Tracking()
                    Tracking.trackings[current_track.id] = current_track

                # Prepara os dados da captura para adicionar ao tracking.
                capture_data = {
                    'input_frame': frame, 
                    'bounding_box': [x1, y1, x2, y2], 
                    'input_name': input_name, 
                    'reading': plate_text
                }

                # Adiciona a captura ao objeto de tracking atual.
                current_track.addCapture(capture_data)
        else:
            # Se nenhuma placa for detectada, loga a informação (se mudou de estado).
            if last_plate_count != 0:
                logging.info("Nenhuma placa detectada neste frame.")
                last_plate_count = 0

        # --- Cálculo e Exibição de FPS ---
        if settings.CALCULATE_FPS:
            fps_frame_count += 1
            # Calcula o FPS a cada segundo.
            if (time.time() - fps_start_time) > 1.0:
                fps = fps_frame_count / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0

            # Loga o FPS de forma inteligente, apenas se a variação for significativa.
            if abs(fps - last_logged_fps) > settings.FPS_LOG_LIMIAR:
                logging.info(f"FPS: {fps:.2f}")
                last_logged_fps = fps

        # --- Interface de Visualização (se habilitada) ---
        if settings.SHOW_CAPTURES:
            display_frame = frame.copy()

            # Desenha os polígonos de máscara na imagem de exibição.
            if polygons:
                polygon_pts = [np.array(p, dtype=np.int32) for p in polygons]
                cv2.polylines(display_frame, polygon_pts, isClosed=True, color=(0, 255, 0), thickness=2)

            # Exibe o valor de FPS na tela.
            if settings.CALCULATE_FPS:
                fps_text = f"FPS: {fps:.2f}"
                cv2.putText(display_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Exibe informações do tracking atual.
            if current_track and current_track.id in Tracking.trackings:
                info_y = 70
                cv2.putText(display_frame, f"Tracking ID: {current_track.id[:8]}", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display_frame, f"Leituras: {len(current_track.readings)}", 
                           (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                if current_track.possibleReadings:
                    cv2.putText(display_frame, f"Melhor: {current_track.possibleReadings[0]}", 
                               (10, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Exibe o número de trackings ativos.
            active_count = len(Tracking.trackings)
            cv2.putText(display_frame, f"Ativos: {active_count}", 
                       (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Mostra o frame processado em uma janela.
            cv2.imshow(f"Frame - {input_name}", display_frame)
            # Encerra o loop se a tecla 'q' for pressionada.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logging.info(f"[{input_name}] Tecla 'q' pressionada, encerrando captura.")
                break
    
    # --- Limpeza e Finalização ---
    logging.info(f"[{input_name}] Finalizando processamento. Aguardando fechamento de trackings pendentes...")
    
    # Força o fechamento de todos os trackings que ainda estiverem ativos.
    for track_id in list(Tracking.trackings.keys()):
        track = Tracking.trackings.get(track_id)
        if track and not track.closing:
            logging.info(f"[{input_name}] Forçando fechamento do tracking {track_id}")
            track._start_async_close()
    
    # Aguarda um tempo para que os fechamentos assíncronos sejam concluídos.
    max_wait_time = 30  # segundos
    wait_start = time.time()
    
    while Tracking.trackings and (time.time() - wait_start) < max_wait_time:
        active_count = len(Tracking.trackings)
        if active_count > 0:
            logging.info(f"[{input_name}] Aguardando: {active_count} trackings ativos.")
            time.sleep(1)
        else:
            break
    
    # Libera os recursos de captura de vídeo e fecha as janelas do OpenCV.
    video_source.release()
    cv2.destroyAllWindows()
    logging.info(f"[{input_name}] Processamento finalizado.")