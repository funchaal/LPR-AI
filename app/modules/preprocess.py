import cv2
import numpy as np

def draw_polygonal_mask(frame, polygons):
    """
    Desenha polígonos pretos diretamente sobre o frame para ocultar áreas.
    """
    if not polygons:
        return frame

    # É importante trabalhar em uma cópia para não alterar o frame original
    # que pode ser usado em outras partes do seu loop.
    processed_frame = frame.copy()

    # Converte os pontos do polígono
    polygon_pts = [np.array(p, dtype=np.int32) for p in polygons]

    # Desenha os polígonos preenchidos com preto (0, 0, 0) diretamente na imagem
    # Nota: A cor é (0, 0, 0) porque o frame tem 3 canais de cor (BGR).
    cv2.fillPoly(processed_frame, polygon_pts, (0, 0, 0))

    return processed_frame