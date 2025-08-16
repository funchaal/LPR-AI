@echo off
REM ===========================
REM Script para criar/ativar venv e rodar main.py
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

    REM Instala dependências apenas na primeira vez
    if exist requirements.txt (
        echo Instalando dependencias...
        pip install -r requirements.txt
    )
) else (
    REM Ativa a venv já existente
    call "%VENV_DIR%\Scripts\activate.bat"
)

REM Roda o main.py
echo Rodando o projeto...
python app\main.py

pause