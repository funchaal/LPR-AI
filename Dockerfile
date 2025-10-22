# Use a imagem base da NVIDIA com CUDA e cuDNN (versão runtime é menor e ideal para produção)
FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04

# Evita que instaladores peçam inputs interativos
ENV DEBIAN_FRONTEND=noninteractive

# Instala Python, pip e dependências essenciais para OpenCV
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho raiz para o projeto dentro do contêiner
WORKDIR /workspace

# Copia todo o conteúdo do diretório de build local (platetrack/) para o WORKDIR (/workspace)
# Isso inclui a pasta 'app', 'requirements.txt', 'config.json', etc.
COPY . .

# Instala as dependências Python a partir do requirements.txt E o TensorRT
# Combinar em um único RUN é mais eficiente para o cache de camadas
RUN pip3 install --no-cache-dir -r requirements.txt \
    tensorrt-cu12>=7.0.0,!=10.1.0

# Define o comando que será executado, ajustando o caminho para main.py
CMD ["python3", "app/main.py"]
