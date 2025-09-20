setlocal EnableExtensions EnableDelayedExpansion
cls
title Configurador e Executor do Projeto de IA (CPU-Only)

REM =================================================================================
REM SCRIPT DE CONFIGURACAO E EXECUCAO v3.1 (CPU-Only)
REM
REM Script ajustado para instalar dependencias de inferencia apenas para CPU.
REM Baseado em PaddlePaddle 3.0 e PaddleOCR 3.2 com PyTorch 2.6.0.
REM =================================================================================

REM Garante que o script esta executando a partir de seu proprio diretorio
pushd "%~dp0"

set "VENV_DIR=venv"
set "PYTHON_MAIN_SCRIPT=app\main.py"
set "REQUIREMENTS_FILE=requirements.txt"

REM --- PASSO PRELIMINAR: Verificar se o Python esta instalado e no PATH ---
python --version >nul 2>&1
if errorlevel 1 (
    echo ------------------------------------------------------------------
    echo   ERRO: O comando 'python' nao foi encontrado no sistema.
    echo.
    echo   Por favor, instale o Python 3.9+ e certifique-se de marcar
    echo   a opcao "Add Python to PATH" durante a instalacao.
    echo.
    echo   Baixe em: https://www.python.org/downloads/
    echo ------------------------------------------------------------------
    pause
    popd
    exit /b 1
)

REM --- PASSO 1: Verificar se o ambiente virtual precisa ser criado ---
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo.
    echo ------------------------------------------------------------------
    echo   Ambiente virtual '%VENV_DIR%' nao encontrado.
    echo   Iniciando o assistente de configuracao (CPU-Only)...
    echo ------------------------------------------------------------------
    echo.

    REM --- PASSO 2: Criar o ambiente virtual ---
    echo [1/5] Criando ambiente virtual...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 ( echo ERRO: Nao foi possivel criar o ambiente virtual. & pause & popd & exit /b 1 )
    call "%VENV_DIR%\Scripts\activate.bat"

    REM --- PASSO 3: Instalar frameworks para CPU ---
    echo.
    echo [2/5] Instalando frameworks de IA para CPU. Isso pode demorar...
    python -m pip install --upgrade pip >nul
    pip install setuptools wheel
    if errorlevel 1 ( echo ERRO ao instalar setuptools. & pause & popd & exit /b 1 )
    
    echo      Instalando PyTorch 2.6.0 (CPU)...
    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 ( echo ERRO ao instalar PyTorch. & pause & popd & exit /b 1 )
    
    echo      Instalando PaddlePaddle 3.0.0 (CPU)...
    python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
    if errorlevel 1 ( echo ERRO ao instalar PaddlePaddle. & pause & popd & exit /b 1 )
    
    REM --- PASSO 4: Instalar PaddleOCR ---
    echo.
    echo [3/5] Instalando PaddleOCR 3.2...
    pip install paddleocr==3.2.0
    if errorlevel 1 ( echo ERRO ao instalar PaddleOCR 3.2. & pause & popd & exit /b 1 )

    REM --- PASSO 5: Instalar o resto das dependencias ---
    echo.
    echo [4/5] Instalando dependencias comuns de %REQUIREMENTS_FILE%...
    pip install -r "%REQUIREMENTS_FILE%"
    if errorlevel 1 ( echo ERRO ao instalar dependencias de %REQUIREMENTS_FILE%. & pause & popd & exit /b 1 )

    echo.
    echo ------------------------------------------------------------------
    echo   Configuracao para CPU concluida!
    echo ------------------------------------------------------------------
    echo.
    echo [5/5] O ambiente esta pronto. Execute o script novamente para iniciar o programa.
    pause
    popd
    exit /b 0
)

REM --- Ativa o ambiente e executa o programa principal ---
echo Ativando ambiente virtual...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 ( echo ERRO: Nao foi possivel ativar o ambiente virtual. & pause & popd & exit /b 1 )

echo.
echo Iniciando o script principal ^(%PYTHON_MAIN_SCRIPT%^)...
echo Usando PaddlePaddle 3.0 + PaddleOCR 3.2 + PyTorch 2.6.0 (Modo CPU)
echo.

if not exist "%PYTHON_MAIN_SCRIPT%" (
    echo ERRO: Nao foi possivel encontrar o script principal: '%PYTHON_MAIN_SCRIPT%'
    pause
    popd
    exit /b 1
)

python "%PYTHON_MAIN_SCRIPT%"
if errorlevel 1 (
    echo.
    echo ------------------------------------------------------------------
    echo   ERRO: O script Python encontrou um problema.
    echo   Verifique a saida de erro acima para mais detalhes.
    echo ------------------------------------------------------------------
) else (
    echo.
    echo Programa finalizado com sucesso.
)

echo Pressione qualquer tecla para sair.
pause
popd
endlocal
