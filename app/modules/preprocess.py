import cv2
import numpy as np
import os
import time

def draw_polygonal_mask(frame, polygons, save_frame=False, save_dir="frames"):
    """
    Desenha polígonos pretos diretamente sobre o frame para ocultar áreas.
    Se save_frame=True, salva o resultado em disco.
    """
    if not polygons:
        processed_frame = frame.copy()
    else:
        processed_frame = frame.copy()
        # Converte os pontos do polígono
        polygon_pts = [np.array(p, dtype=np.int32) for p in polygons]
        # Desenha os polígonos preenchidos com preto
        cv2.fillPoly(processed_frame, polygon_pts, (0, 0, 0))

    if save_frame:
        # Cria diretório de saída se não existir
        os.makedirs(save_dir, exist_ok=True)

        # Nome do arquivo com timestamp
        filename = os.path.join(save_dir, f"frame_{int(time.time() * 1000)}.jpg")

        # Salva o frame
        cv2.imwrite(filename, processed_frame)

    return processed_frame
