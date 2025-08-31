import cv2
import numpy as np

def draw_polygonal_mask(frame, polygons):
    """
    Aplica uma máscara poligonal a um frame. Apenas as áreas dentro dos polígonos
    permanecerão visíveis.
    """
    if not polygons:
        return frame

    # Cria uma máscara preta com as mesmas dimensões do frame
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Converte os pontos do polígono para o formato que o OpenCV necessita
    polygon_pts = [np.array(p, dtype=np.int32) for p in polygons]

    # Preenche os polígonos na máscara com a cor branca
    cv2.fillPoly(mask, polygon_pts, (255))

    # Aplica a máscara ao frame original usando uma operação bitwise_and
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    return masked_frame