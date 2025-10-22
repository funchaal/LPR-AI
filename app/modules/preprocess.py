import cv2
import numpy as np
import os
import time

def draw_polygonal_mask(frame, polygons, save_frame=False, save_dir="frames"):
    """
    Cria uma máscara para tornar visível apenas a área dentro dos polígonos,
    deixando o resto da imagem em preto.
    Se a lista de polígonos estiver vazia, retorna o frame original.
    Se save_frame=True, salva o resultado em disco.
    """
    # Se não houver polígonos, retorna o frame original imediatamente
    if not polygons:
        return frame

    # Cria uma máscara preta com as mesmas dimensões do frame
    mask = np.zeros_like(frame, dtype=np.uint8)

    # Converte os pontos dos polígonos para o formato que o OpenCV espera
    polygon_pts = [np.array(p, dtype=np.int32) for p in polygons]
    
    # Desenha os polígonos na máscara, mas com preenchimento branco (255)
    cv2.fillPoly(mask, polygon_pts, (255, 255, 255))
    
    # Usa a máscara para manter apenas a área dentro dos polígonos do frame original
    # A operação bitwise_and mantém os pixels do frame onde a máscara é branca
    processed_frame = cv2.bitwise_and(frame, mask)

    if save_frame:
        # Cria o diretório de saída se ele não existir
        os.makedirs(save_dir, exist_ok=True)
        
        # Gera um nome de arquivo único usando timestamp
        filename = os.path.join(save_dir, f"frame_{int(time.time() * 1000)}.jpg")
        
        # Salva o frame processado
        cv2.imwrite(filename, processed_frame)

    return processed_frame

def resize_with_padding(image, target_height, target_width):
    """
    Resizes an image to a target height while maintaining the aspect ratio,
    and then adds black padding to meet the target width.

    The process is as follows:
    1. The image is resized to the `target_height`, calculating the
       corresponding width to maintain the original aspect ratio.
    2. If this new width is less than the `target_width`, the image is
       centered and padded with black borders on the sides.
    3. If the new width is greater than the `target_width` (which happens
       with very wide images), the image is resized directly to the target
       dimensions, which may distort the aspect ratio.

    Args:
        image (np.ndarray): The input image (e.g., a video frame).
        target_height (int): The final desired height for the image.
        target_width (int): The final desired width for the image.

    Returns:
        np.ndarray: The processed image with the final desired dimensions.
    """
    # Get the original image dimensions
    original_height, original_width = image.shape[:2]

    # Calculate the new width to maintain the aspect ratio based on the new height
    ratio = target_height / float(original_height)
    new_width = int(original_width * ratio)

    # Resize the image to the new height and calculated width.
    # The INTER_AREA interpolation method is efficient for image shrinking.
    resized_image = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_AREA)

    # Check if padding is needed
    if new_width < target_width:
        # Calculate the total padding needed
        total_padding = target_width - new_width
        
        # Split the padding for left and right to center the image
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding

        # Add the black borders (top, bottom, left, right)
        final_image = cv2.copyMakeBorder(
            resized_image,
            0,                 # No top padding
            0,                 # No bottom padding
            left_padding,
            right_padding,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)    # Border color: black
        )
    # If the calculated width is greater, resize directly to the target (distorting)
    elif new_width > target_width:
        final_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    # If the width is the same, do nothing
    else:
        final_image = resized_image

    return final_image