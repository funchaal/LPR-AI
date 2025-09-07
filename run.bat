@echo off
setlocal EnableExtensions EnableDelayedExpansion
cls
title Configurador e Executor do Projeto de IA

REM =================================================================================
REM SCRIPT DE CONFIGURACAO E EXECUCAO v2.5 (fix: URL de instalacao do Paddle)
REM
REM Corrigido o comando de instalacao do PaddlePaddle para usar a URL
REM oficial, garantindo que a versao correta para CUDA 12 seja encontrada.
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
    echo [1/5] Criando ambiente virtual...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 ( echo ERRO: Nao foi possivel criar o ambiente virtual. & pause & popd & exit /b 1 )
    call "%VENV_DIR%\Scripts\activate.bat"

    REM --- PASSO 3: Perguntar sobre GPU e instalar frameworks ---
    echo.
    echo [2/5] Selecao do ambiente de execucao.
    CHOICE /C SN /N /M "Esta maquina possui uma GPU NVIDIA com o CUDA Toolkit ^(versao 12.x^) instalado? [S/N]: "

    if errorlevel 2 (
        REM --- ROTA DE INSTALACAO PARA CPU ---
        echo.
        echo    Opcao selecionada: CPU. Instalando...
        python -m pip install --upgrade pip >nul
        pip install setuptools
        if errorlevel 1 ( echo ERRO ao instalar setuptools. & pause & popd & exit /b 1 )
        pip install torch==2.3.1 torchvision==0.18.1
        if errorlevel 1 ( echo ERRO ao instalar PyTorch. & pause & popd & exit /b 1 )
        pip install paddlepaddle==2.6.1
        if errorlevel 1 ( echo ERRO ao instalar PaddlePaddle. & pause & popd & exit /b 1 )
    ) else (
        REM --- ROTA DE INSTALACAO PARA GPU ---
        echo.
        echo    Opcao selecionada: GPU. Instalando PyTorch, PaddlePaddle e cuDNN ^(pode demorar^)...
        python -m pip install --upgrade pip >nul
	pip install setuptools
        if errorlevel 1 ( echo ERRO ao instalar setuptools. & pause & popd & exit /b 1 )
        pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
        if errorlevel 1 ( echo ERRO ao instalar PyTorch para GPU. & pause & popd & exit /b 1 )

        REM AQUI ESTA A CORRECAO: Usando a URL oficial do Paddle para encontrar a versao de CUDA 12
        python -m pip install paddlepaddle-gpu==3.1.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
        if errorlevel 1 ( echo ERRO ao instalar PaddlePaddle-GPU. & pause & popd & exit /b 1 )

        pip install nvidia-cudnn-cu12
        if errorlevel 1 ( echo ERRO ao instalar cuDNN. & pause & popd & exit /b 1 )

        REM --- PASSO 4: Criar e executar script de verificacao ---
        echo.
        echo [3/5] Verificando se o ambiente CUDA esta funcionando corretamente...
        (
            echo import torch
            echo import paddle
            echo import sys
            echo.
            echo print^("--- Verificando o PyTorch ---"^)
            echo success = True
            echo if not torch.cuda.is_available^(^):
            echo     print^("ERRO: PyTorch NAO conseguiu detectar a GPU com CUDA."^)
            echo     success = False
            echo else:
            echo     print^("SUCESSO: PyTorch detectou a GPU: %%s" %% torch.cuda.get_device_name^(0^)^)
            echo.
            echo print^("\n--- Verificando o PaddlePaddle ---"^)
            echo if not paddle.is_compiled_with_cuda^(^):
            echo     print^("ERRO: A versao do PaddlePaddle instalada NAO e compativel com GPU."^)
            echo     success = False
            echo else:
            echo     try:
            echo         paddle.utils.run_check^(^)
            echo         print^("SUCESSO: PaddlePaddle esta comunicando com o hardware."^)
            echo     except Exception as e:
            echo         print^("ERRO: PaddlePaddle FALHOU ao comunicar com o hardware."^)
            echo         success = False
            echo.
            echo if not success:
            echo     print^("\n--- DIAGNOSTICO ---"^)
            echo     print^("O ambiente Python nao conseguiu usar a GPU, apesar do CUDA Toolkit estar instalado."^)
            echo     print^("CAUSA MAIS COMUM: O seu DRIVER DA NVIDIA esta desatualizado."^)
            echo     print^("SOLUCAO: Atualize o driver da sua GPU para a versao mais recente no site da NVIDIA e tente a instalacao novamente."^)
            echo     sys.exit^(1^)
            echo else:
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

    REM --- PASSO 5: Instalar o resto das dependencias ---
    echo.
    echo [4/5] Instalando dependencias comuns de %REQUIREMENTS_FILE%...
    pip install -r "%REQUIREMENTS_FILE%"
    if errorlevel 1 ( echo ERRO ao instalar dependencias de %REQUIREMENTS_FILE%. & pause & popd & exit /b 1 )

    echo.
    echo ------------------------------------------------------------------
    echo   Configuracao concluida com sucesso!
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