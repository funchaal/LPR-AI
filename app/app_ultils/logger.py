import os
import logging

def setup_logger(PATH):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(PATH, "plate_recognition.log"), mode='a'),
            logging.StreamHandler()
        ], 
        force=True
    )
