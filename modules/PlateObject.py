from collections import Counter, defaultdict
import cv2
import sqlite3
from datetime import datetime
import logging
import json
import uuid
import os
from datetime import datetime
import cv2
from numpy import save

class PlateObject:

    limiar = 1.5
    instances = {}
    suspect_detections = []
    suspect_detections_save_path = None
    api = lambda instance, readings: None  # Placeholder for API function

    def __init__(self, instance_id):
        self.start_time = datetime.now()
        self.id = str(uuid.uuid4())
        self.instanceId = instance_id
        self.closed = False
        self.readings = defaultdict(int)
        self.possibleReadings = []
        self.finalReading = ''
        self.noFrameCount = 0
        self.frames = []
        
    
    def __str__(self):
        return f'({self.id}) possible plates: {self.possibleReadings}, final plate: {self.finalReading}'
    
    @classmethod
    def setup(cls, db_connection, captures_save_path, suspect_detections_save_path, api=lambda instance, readings: None):
        cls.db_connection = sqlite3.connect(db_connection)
        cls.suspect_detections_save_path = suspect_detections_save_path
        cls.captures_save_path = captures_save_path
        cls.api = api
        cursor = cls.db_connection.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS placas (
                id TEXT PRIMARY KEY,
                instance_id TEXT,
                final_plate TEXT,
                timestamp TEXT,
                imagem_path TEXT,
                readings_json TEXT,
                possible_plates_json TEXT
            )
        ''')
        cls.db_connection.commit()

    @classmethod
    def newFrame(cls, instance_id):
        instance = cls.instances.get(instance_id)
        if not instance:
            return

        instance.noFrameCount += 1

        if instance.noFrameCount > 5:
            instance.closed = True
            logging.info(f"instance {instance_id} closed due to inactivity")
            instance.finalReading = instance.definePossibleReadings(instance.readings)[0]

            print("📸 Leituras registradas até o momento:")
            print(instance.readings)

            print("🔍 Placas possíveis identificadas:")
            print(instance.possibleReadings)

            print("✅ Placa final escolhida para o instanceo:")
            print(instance.finalReading)

            print('timestamp: ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            duration = datetime.now() - instance.start_time
            print(f"⏱️ instanceamento da placa {instance.finalReading} levou {duration.total_seconds():.2f} segundos.")

            instance.close()

    @classmethod
    def save_suspect_detections(cls):
        if not cls.suspect_detections:
            return
        
        os.makedirs(cls.suspect_detections_save_path, exist_ok=True)

        for detection in cls.suspect_detections:
            frame_id = detection["frame_id"]
            frame = detection["frame"]
            x1, y1, x2, y2 = detection["coords"]
            tipo = detection["type"]

            # Recorta a imagem com base nas coordenadas
            cropped = frame[y1:y2, x1:x2]

            # Formata o nome do arquivo
            filename = f"{frame_id} {x1}-{y1}-{x2}-{y2} {tipo}.jpg"
            filepath = os.path.join(cls.suspect_detections_save_path, filename)

            # Salva a imagem
            cv2.imwrite(filepath, cropped)

        cls.suspect_detections = []
        
        logging.info(f"Detecções suspeitas salvas com sucesso.")

    def close(self):
        logging.info(f"Placa {self.id} finalizada com leitura: {self.finalReading}")

        best_frame = self.chooseBestFrame(self.frames)
        captures_save_path = self.__class__.captures_save_path

        try:
            self.__class__.save_suspect_detections()
        except Exception as e:
            logging.error(f"Erro ao salvar detecções suspeitas: {e}")

        try:
            if captures_save_path is not None:
                # Pegando a pasta pai do captures_save_path
                base_folder = os.path.dirname(captures_save_path)

                # Data atual
                now = datetime.now()
                year = str(now.year)
                month = f"{now.month:02d}"  # mês com zero à esquerda
                day = f"{now.day:02d}"

                # Monta o caminho completo: base_folder/ano/mes/dia
                folder_path = os.path.join(base_folder, year, month, day)

                # Cria as pastas, se não existirem
                os.makedirs(folder_path, exist_ok=True)

                # Nome do arquivo a partir do captures_save_path original
                filename = os.path.basename(f"{self.id}.jpg")

                # Caminho final com pasta ano/mes/dia
                final_path = os.path.join(folder_path, filename)

                # Salva a imagem no caminho final
                cv2.imwrite(final_path, best_frame)

                logging.info(f"Captura salva para placa {self.finalReading}")
        except Exception as e:
            logging.error(f"Erro ao salvar captura: {e}")

        try:
            cursor = self.__class__.db_connection.cursor()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            imagem_path = f"{self.id}.jpg"

            readings_json = json.dumps(self.readings)  # lista de tuplas
            possible_plates_json = json.dumps(self.possibleReadings)

            cursor.execute('''
                INSERT OR REPLACE INTO placas 
                (id, instance_id, final_plate, timestamp, imagem_path, readings_json, possible_plates_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.id,
                self.instanceId,
                self.finalReading,
                timestamp,
                imagem_path,
                readings_json,
                possible_plates_json
            ))

            self.__class__.db_connection.commit()
            logging.info(f"Dados salvos no SQLite para placa {self.finalReading}")

        except Exception as e:
            logging.error(f"Erro ao salvar no banco SQLite: {e}")

        finally:
            del self.__class__.instances[self.instanceId]

    
    def addCapture(self, reading, frame):
        self.readings[reading] += 1
        self.frames.append(frame)

        self.noFrameCount = 0

        if all(len(item) < 7 for item in self.readings):
            logging.debug(f"Leitura {reading} ignorada por ser muito curta.")
            return

        defined_possible_plates = self.definePossibleReadings(self.readings)
        new_possible_plates = [[idx, x] for idx, x in enumerate(defined_possible_plates) if x not in self.possibleReadings]
        
        if new_possible_plates:
            for idx, plate in new_possible_plates:
                self.possibleReadings.insert(idx, plate)

            self.__class__.api(instance=self.instanceId, readings=[plate for idx, plate in new_possible_plates])

    def chooseBestFrame(self, frames):
        if not frames:
            return None

        best_frame = None
        min_distance = float('inf')

        for frame in frames:
            x1, y1, x2, y2 = frame['plateBoundingBox']
            bbox_center_x = (x1 + x2) / 2
            bbox_center_y = (y1 + y2) / 2

            height, width = frame['inputFrame'].shape[:2]
            image_center_x = width / 2
            image_center_y = height / 2

            distance = ((bbox_center_x - image_center_x) ** 2 + (bbox_center_y - image_center_y) ** 2) ** 0.5

            if distance < min_distance:
                min_distance = distance
                best_frame = frame

        if best_frame is None:
            return None

        plate_frame = best_frame['inputFrame'].copy()
        x1, y1, x2, y2 = best_frame['plateBoundingBox']

        # Desenha a bounding box
        cv2.rectangle(plate_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text = self.finalReading if self.finalReading else "N/A"
        text_pos = (x1, max(y1 - 10, 10))

        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(
            plate_frame,
            (text_pos[0], text_pos[1] - text_height - baseline),
            (text_pos[0] + text_width, text_pos[1] + baseline),
            (0, 255, 0), cv2.FILLED
        )

        cv2.putText(plate_frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return plate_frame


    def definePossibleReadings(self, plates):
        plate_pontuation = defaultdict(int)

        for plate in plates.keys():
            substrings = set(
                plate[j:j + i]
                for i in range(2, len(plate))
                for j in range(len(plate) - i + 1)
            )
            for substring in substrings:
                for reading, count in self.readings.items():
                    if substring in reading:
                        plate_pontuation[plate] += count

        top_plates = sorted(plate_pontuation, key=plate_pontuation.get, reverse=True)[:2]
        return top_plates
