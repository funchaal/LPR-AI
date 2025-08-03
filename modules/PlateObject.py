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
    processes = {}
    api = lambda process, readings: None  # Placeholder for API function

    def __init__(self, process_id):
        self.start_time = datetime.now()
        self.id = str(uuid.uuid4())
        self.processId = process_id
        self.closed = False
        self.readings = defaultdict(int)
        self.possibleReadings = []
        self.finalReading = ''
        self.noFrameCount = 0
        self.frames = []
        
    
    def __str__(self):
        return f'({self.id}) possible plates: {self.possibleReadings}, final plate: {self.finalReading}'
    
    @classmethod
    def setup(cls, db_connection, captures_save_path, api=lambda process, readings: None):
        cls.db_connection = sqlite3.connect(db_connection)
        cls.captures_save_path = captures_save_path
        cls.api = api
        cursor = cls.db_connection.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS placas (
                id TEXT PRIMARY KEY,
                process_id TEXT,
                final_plate TEXT,
                timestamp TEXT,
                imagem_path TEXT,
                readings_json TEXT,
                possible_plates_json TEXT
            )
        ''')
        cls.db_connection.commit()

    @classmethod
    def newFrame(cls, process_id):
        process = cls.processes.get(process_id)
        if not process:
            return

        process.noFrameCount += 1

        if process.noFrameCount > 5:
            process.closed = True
            logging.info(f"Process {process_id} closed due to inactivity")
            process.finalReading = process.definePossibleReadings(process.readings)[0]

            print("📸 Leituras registradas até o momento:")
            print(process.readings)

            print("🔍 Placas possíveis identificadas:")
            print(process.possibleReadings)

            print("✅ Placa final escolhida para o processo:")
            print(process.finalReading)

            print('timestamp: ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            duration = datetime.now() - process.start_time
            print(f"⏱️ Processamento da placa {process.finalReading} levou {duration.total_seconds():.2f} segundos.")

            process.close()


    def close(self):
        self.chooseBestFrame(self.frames)
        logging.info(f"Placa {self.id} finalizada com leitura: {self.finalReading}")

        try:
            cursor = self.__class__.db_connection.cursor()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            imagem_path = f"{self.id}.jpg"

            readings_json = json.dumps(self.readings)  # lista de tuplas
            possible_plates_json = json.dumps(self.possibleReadings)

            cursor.execute('''
                INSERT OR REPLACE INTO placas 
                (id, process_id, final_plate, timestamp, imagem_path, readings_json, possible_plates_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.id,
                self.processId,
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
            del self.__class__.processes[self.processId]

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

            self.__class__.api(process=self.processId, readings=[plate for idx, plate in new_possible_plates])

    def chooseBestFrame(self, frames):
        if not frames:
            return None
        
        save_path = self.__class__.captures_save_path

        mid_index = len(frames) // 2
        central_frame = frames[mid_index]

        plate_frame = central_frame['inputFrame'].copy()  # copia para não alterar o original
        x1, y1, x2, y2 = central_frame['plateBoundingBox']

        # Desenha a bounding box na imagem
        cv2.rectangle(plate_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Texto para colocar (a placa final)
        text = self.finalReading if self.finalReading else "N/A"

        # Posição do texto: acima da caixa
        text_pos = (x1, max(y1 - 10, 10))

        # Desenha o retângulo de fundo do texto para melhorar a visibilidade
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(plate_frame, 
                    (text_pos[0], text_pos[1] - text_height - baseline),
                    (text_pos[0] + text_width, text_pos[1] + baseline), 
                    (0, 255, 0), cv2.FILLED)

        # Escreve o texto
        cv2.putText(plate_frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        if save_path is not None:
            # Pegando a pasta pai do save_path
            base_folder = os.path.dirname(save_path)

            # Data atual
            now = datetime.now()
            year = str(now.year)
            month = f"{now.month:02d}"  # mês com zero à esquerda
            day = f"{now.day:02d}"

            # Monta o caminho completo: base_folder/ano/mes/dia
            folder_path = os.path.join(base_folder, year, month, day)

            # Cria as pastas, se não existirem
            os.makedirs(folder_path, exist_ok=True)

            # Nome do arquivo a partir do save_path original
            filename = os.path.basename(f"{self.id}.jpg")

            # Caminho final com pasta ano/mes/dia
            final_path = os.path.join(folder_path, filename)

            # Salva a imagem no caminho final
            cv2.imwrite(final_path, plate_frame)

        return

    def definePossibleReadings(self, plates):
        plate_pontuation = defaultdict(int)
        for plate in plates.keys():
            for i in range(2, len(plate)):
                for j in range(len(plate) - i + 1):
                    substring = plate[j:j + i]
                    for reading, count in self.readings.items():
                        if substring in reading:
                            plate_pontuation[plate] += count
        
        top_plates = sorted(plate_pontuation, key=plate_pontuation.get, reverse=True)[:2]

        return top_plates
