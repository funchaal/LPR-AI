import sqlite3
import logging
import json
from datetime import datetime

class CapturesDatabase:
    """Gerencia todas as operações com o banco de dados de placas."""

    def __init__(self, db_path: str):
        """Inicializa a conexão com o banco e cria a tabela se não existir."""
        try:
            self.db_connection = sqlite3.connect(db_path, check_same_thread=False)
            self._create_table()
        except sqlite3.Error as e:
            logging.error(f"Erro ao conectar ou criar tabela no banco de dados: {e}")
            raise

    def _create_table(self):
        """Cria a tabela 'placas' se ela ainda não existir."""
        cursor = self.db_connection.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS placas (
                id TEXT PRIMARY KEY,
                instance_id TEXT,
                final_reading TEXT,
                timestamp TEXT,
                imagem_path TEXT,
                readings_json TEXT,
                possible_plates_json TEXT
            )
        ''')
        self.db_connection.commit()

    def save_tracking(self, tracking_data: dict):
        """Salva os dados de um rastreamento finalizado no banco."""
        try:
            cursor = self.db_connection.cursor()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Usando .get() para segurança caso uma chave não exista
            readings_json = json.dumps(tracking_data.get('readings', {}))
            possible_plates_json = json.dumps(tracking_data.get('possibleReadings', []))

            cursor.execute('''
                INSERT OR REPLACE INTO placas 
                (id, instance_id, final_reading, timestamp, imagem_path, readings_json, possible_plates_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                tracking_data.get('id'),
                tracking_data.get('instance_id'),
                tracking_data.get('finalReading'),
                timestamp,
                tracking_data.get('image_path'),
                readings_json,
                possible_plates_json
            ))
            self.db_connection.commit()
            logging.info(f"Dados salvos no SQLite para a passagem {tracking_data.get('id')}")
        except sqlite3.Error as e:
            logging.error(f"Erro ao salvar no banco SQLite: {e}")

    def close(self):
        """Fecha a conexão com o banco de dados."""
        if self.db_connection:
            self.db_connection.close()
            logging.info("Conexão com o banco de dados fechada.")