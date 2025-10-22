# Use a imagem base da NVIDIA com CUDA e cuDNN (versão runtime é menor e ideal para produção)
FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04

# Evita que instaladores peçam inputs interativos
ENV DEBIAN_FRONTEND=noninteractive

# Instala Python, pip e dependências essenciais para OpenCV, incluindo as de GUI (Qt5)
# Esta linha foi corrigida para remover espaços inválidos.
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libqt5gui5 \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho raiz para o projeto dentro do contêiner
WORKDIR /workspace

# Copia todo o conteúdo do diretório de build local (platetrack/) para o WORKDIR (/workspace)
# Isso inclui a pasta 'app', 'requirements.txt', 'config.json', etc.
COPY . .

# Instala as dependências Python a partir do requirements.txt que foi copiado
RUN pip3 install --no-cache-dir -r requirements.txt \
    tensorrt-cu12>=7.0.0,!=10.1.0

# Expõe a porta, caso sua aplicação tenha um servidor web (opcional, mas boa prática)
# EXPOSE 8000

# Define o comando que será executado, ajustando o caminho para main.py
CMD ["python3", "app/main.py"]