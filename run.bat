@echo off
REM ===========================
REM Script baseado no SEU que funciona
REM Apenas adicionando instalacoes de IA
REM ===========================

set VENV_DIR=venv

REM Verifica se a venv existe
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Criando virtualenv...
    python -m venv %VENV_DIR%
    
    REM Ativa a venv recem-criada
    call "%VENV_DIR%\Scripts\activate.bat"
    
    REM Atualiza pip
    python -m pip install --upgrade pip
    
    REM Instala dependencias de IA primeiro
    echo Instalando PyTorch CPU...
    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
    
    echo Instalando PaddlePaddle CPU...
    python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
    
    echo Instalando PaddleOCR...
    pip install paddleocr==3.2
    
    REM Instala dependências apenas na primeira vez
    if exist requirements.txt (
        echo Instalando dependencias...
        pip install -r requirements.txt
    )
    
    echo.
    echo Instalacao concluida! Execute novamente para rodar o programa.
    pause
    exit /b 0
    
) else (
    REM Ativa a venv já existente
    call "%VENV_DIR%\Scripts\activate.bat"
)

REM Roda o main.py
echo Rodando o projeto...
python app\main.py

pause