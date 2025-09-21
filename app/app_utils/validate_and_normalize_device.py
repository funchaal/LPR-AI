import logging
import torch

def validate_and_normalize_device(requested_device: str) -> str:
    """
    Valida e normaliza um dispositivo solicitado ('cpu', 'gpu:0', etc.).
    Se a GPU solicitada não estiver disponível, tenta GPUs anteriores ou cai em 'cpu'.

    Args:
        requested_device (str): O dispositivo solicitado.

    Returns:
        str: Dispositivo validado e normalizado ('cpu' ou 'gpu:X').
    """
    device_str = requested_device.lower().strip()

    # Se não for GPU, retorna 'cpu' diretamente
    if not any(k in device_str for k in ['cuda', 'gpu']):
        logging.debug(f"Dispositivo '{requested_device}' validado como 'cpu'.")
        return "cpu"

    # Checa se há suporte a GPU
    is_gpu_available = torch.cuda.is_available()
    is_paddle_gpu_version = False
    try:
        import paddle
        is_paddle_gpu_version = paddle.is_compiled_with_cuda()
    except Exception:
        pass

    logging.debug(f"Validando dispositivo GPU '{requested_device}'...")
    logging.debug(f"  - Hardware (NVIDIA): {'SIM' if is_gpu_available else 'NÃO'}")
    logging.debug(f"  - Software (Paddle GPU): {'SIM' if is_paddle_gpu_version else 'NÃO'}")

    if not (is_gpu_available and is_paddle_gpu_version):
        logging.warning(f"AVISO: GPU não disponível ou Paddle não compilado com CUDA. Revertendo para 'cpu'.")
        return "cpu"

    # Extrai o índice da GPU solicitada
    try:
        gpu_index = int(device_str.split(':')[1]) if ':' in device_str else 0
    except ValueError:
        gpu_index = 0

    # Tenta encontrar uma GPU válida, recuando se necessário
    while gpu_index >= 0:
        if gpu_index < torch.cuda.device_count():
            final_device = f"gpu:{gpu_index}"
            logging.debug(f"Dispositivo '{requested_device}' validado como '{final_device}'.")
            return final_device
        gpu_index -= 1

    logging.warning(f"AVISO: Nenhuma GPU disponível. Usando 'cpu'.")
    return "cpu"