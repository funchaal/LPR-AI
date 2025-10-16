import cv2
import numpy as np
import os
import time

def draw_polygonal_mask(frame, polygons, save_frame=False, save_dir="frames"):
    """
    Cria uma máscara para tornar visível apenas a área dentro dos polígonos,
    deixando o resto da imagem em preto.
    Se save_frame=True, salva o resultado em disco.
    """
    # Cria uma máscara preta com as mesmas dimensões do frame
    mask = np.zeros_like(frame, dtype=np.uint8)

    if polygons:
        # Converte os pontos dos polígonos para o formato que o OpenCV espera
        polygon_pts = [np.array(p, dtype=np.int32) for p in polygons]
        
        # Desenha os polígonos na máscara, mas com preenchimento branco (255)
        cv2.fillPoly(mask, polygon_pts, (255, 255, 255))
        
        # Usa a máscara para manter apenas a área dentro dos polígonos do frame original
        # A operação bitwise_and mantém os pixels do frame onde a máscara é branca
        processed_frame = cv2.bitwise_and(frame, mask)
    else:
        # Se não houver polígonos, retorna um frame todo preto
        processed_frame = mask

    if save_frame:
        # Cria o diretório de saída se ele não existir
        os.makedirs(save_dir, exist_ok=True)
        
        # Gera um nome de arquivo único usando timestamp
        filename = os.path.join(save_dir, f"frame_{int(time.time() * 1000)}.jpg")
        
        # Salva o frame processado
        cv2.imwrite(filename, processed_frame)

    return processed_frame