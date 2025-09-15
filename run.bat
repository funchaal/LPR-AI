@echo off
setlocal EnableExtensions EnableDelayedExpansion
cls
title Configurador e Executor do Projeto de IA

REM =================================================================================
REM SCRIPT DE CONFIGURACAO E EXECUCAO v3.0 (PaddlePaddle 3.0 + PaddleOCR 3.0)
REM
REM Atualizado para usar PaddlePaddle 3.0.0 e PaddleOCR 3.0+ com suporte
REM completo para CUDA 11.8 e compatibilidade com PyTorch 2.6.0
REM =================================================================================

REM Garante que o script esta executando a partir de seu proprio diretorio
pushd "%~dp0"

set "VENV_DIR=venv"
set "PYTHON_MAIN_SCRIPT=app\main.py"
set "REQUIREMENTS_FILE=requirements.txt"
set "VERIFY_SCRIPT=verify_cuda.py"

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
    echo   Iniciando o assistente de configuracao pela primeira vez...
    echo ------------------------------------------------------------------
    echo.

    REM --- PASSO 2: Criar o ambiente virtual ---
    echo [1/6] Criando ambiente virtual...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 ( echo ERRO: Nao foi possivel criar o ambiente virtual. & pause & popd & exit /b 1 )
    call "%VENV_DIR%\Scripts\activate.bat"

    REM --- PASSO 3: Perguntar sobre GPU e instalar frameworks ---
    echo.
    echo [2/6] Selecao do ambiente de execucao.
    echo.
    echo IMPORTANTE: Esta versao requer CUDA 11.8 para GPU. 
    echo Para CUDA 12.x, use apenas CPU por enquanto.
    echo.
    CHOICE /C SN /N /M "Esta maquina possui GPU NVIDIA com CUDA Toolkit 11.8 instalado? [S/N]: "

    if errorlevel 2 (
        REM --- ROTA DE INSTALACAO PARA CPU ---
        echo.
        echo    Opcao selecionada: CPU. Instalando...
        python -m pip install --upgrade pip >nul
        pip install setuptools wheel
        if errorlevel 1 ( echo ERRO ao instalar setuptools. & pause & popd & exit /b 1 )
        
        echo    Instalando PyTorch 2.6.0 CPU...
        pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
        if errorlevel 1 ( echo ERRO ao instalar PyTorch. & pause & popd & exit /b 1 )
        
        echo    Instalando PaddlePaddle 3.0.0 CPU...
        python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
        if errorlevel 1 ( echo ERRO ao instalar PaddlePaddle. & pause & popd & exit /b 1 )
        
    ) else (
        REM --- ROTA DE INSTALACAO PARA GPU ---
        echo.
        echo    Opcao selecionada: GPU. Instalando PyTorch, PaddlePaddle e cuDNN ^(pode demorar^)...
        python -m pip install --upgrade pip >nul
        pip install setuptools wheel
        if errorlevel 1 ( echo ERRO ao instalar setuptools. & pause & popd & exit /b 1 )
        
        echo    Instalando PyTorch 2.6.0 GPU ^(CUDA 11.8^)...
        pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
        if errorlevel 1 ( echo ERRO ao instalar PyTorch para GPU. & pause & popd & exit /b 1 )

        echo    Instalando PaddlePaddle-GPU 3.0.0 ^(CUDA 11.8^)...
        python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
        if errorlevel 1 ( echo ERRO ao instalar PaddlePaddle-GPU. & pause & popd & exit /b 1 )

        echo    Instalando cuDNN para CUDA 11.8...
        pip install nvidia-cudnn-cu11
        if errorlevel 1 ( echo ERRO ao instalar cuDNN. & pause & popd & exit /b 1 )

        REM --- PASSO 4: Criar e executar script de verificacao ---
        echo.
        echo [3/6] Verificando se o ambiente CUDA esta funcionando corretamente...
        (
            echo import torch
            echo import paddle
            echo import sys
            echo.
            echo print^("=== VERIFICACAO DO AMBIENTE GPU ===="^)
            echo print^("PaddlePaddle 3.0.0 + PyTorch 2.6.0 + CUDA 11.8"^)
            echo print^("="*50^)
            echo.
            echo print^("--- Verificando o PyTorch ---"^)
            echo success = True
            echo if not torch.cuda.is_available^(^):
            echo     print^("ERRO: PyTorch NAO conseguiu detectar a GPU com CUDA."^)
            echo     success = False
            echo else:
            echo     device_count = torch.cuda.device_count^(^)
            echo     device_name = torch.cuda.get_device_name^(0^)
            echo     print^(f"SUCESSO: PyTorch detectou {device_count} GPU^(s^): {device_name}"^)
            echo     print^(f"CUDA Version: {torch.version.cuda}"^)
            echo.
            echo print^("--- Verificando o PaddlePaddle ---"^)
            echo if not paddle.is_compiled_with_cuda^(^):
            echo     print^("ERRO: A versao do PaddlePaddle instalada NAO e compativel com GPU."^)
            echo     success = False
            echo else:
            echo     try:
            echo         paddle.utils.run_check^(^)
            echo         print^("SUCESSO: PaddlePaddle 3.0.0 esta comunicando com o hardware."^)
            echo         print^(f"PaddlePaddle version: {paddle.__version__}"^)
            echo     except Exception as e:
            echo         print^(f"ERRO: PaddlePaddle FALHOU ao comunicar com o hardware: {e}"^)
            echo         success = False
            echo.
            echo if not success:
            echo     print^("="*50^)
            echo     print^("DIAGNOSTICO DE PROBLEMAS:"^)
            echo     print^("1. Verifique se o driver NVIDIA esta atualizado ^(versao >=450.80.02^)"^)
            echo     print^("2. Confirme se o CUDA Toolkit 11.8 esta instalado"^)
            echo     print^("3. Reinicie o computador apos instalar drivers"^)
            echo     print^("="*50^)
            echo     sys.exit^(1^)
            echo else:
            echo     print^("="*50^)
            echo     print^("SUCESSO: Ambiente GPU configurado corretamente!"^)
            echo     print^("="*50^)
            echo     sys.exit^(0^)
        ) > %VERIFY_SCRIPT%

        python %VERIFY_SCRIPT%
        if errorlevel 1 (
            echo.
            echo ------------------------------------------------------------------
            echo   FALHA NA VERIFICACAO DO AMBIENTE GPU. Veja as instrucoes acima.
            echo ------------------------------------------------------------------
            del %VERIFY_SCRIPT% >nul 2>&1
            pause
            popd
            exit /b 1
        )
        del %VERIFY_SCRIPT% >nul 2>&1
    )

    REM --- PASSO 5: Instalar PaddleOCR 3.0+ ---
    echo.
    echo [4/6] Instalando PaddleOCR 3.0+...
    pip install paddleocr==3.2.0
    if errorlevel 1 ( echo ERRO ao instalar PaddleOCR 3.0+. & pause & popd & exit /b 1 )

    REM --- PASSO 6: Instalar o resto das dependencias ---
    echo.
    echo [5/6] Instalando dependencias comuns de %REQUIREMENTS_FILE%...
    pip install -r "%REQUIREMENTS_FILE%"
    if errorlevel 1 ( echo ERRO ao instalar dependencias de %REQUIREMENTS_FILE%. & pause & popd & exit /b 1 )

    echo.
    echo ------------------------------------------------------------------
    echo   Configuracao PaddlePaddle 3.0 + PaddleOCR 3.0 concluida!
    echo ------------------------------------------------------------------
    echo.
    echo [6/6] O ambiente esta pronto. Execute o script novamente para iniciar o programa.
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
echo Usando PaddlePaddle 3.0.0 + PaddleOCR 3.0+ + PyTorch 2.6.0
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