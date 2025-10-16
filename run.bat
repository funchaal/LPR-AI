@echo off
setlocal

REM Define o nome do diretório para o ambiente virtual
set "VENV_DIR=venv"

REM Verifica se o Python está disponível
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo Python nao encontrado. Por favor, instale o Python e adicione-o ao PATH do sistema.
    pause
    exit /b 1
)

REM Se a venv NÃO existir, cria e instala tudo.
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Criando o ambiente virtual em '%VENV_DIR%'...
    python -m venv %VENV_DIR%
    if %errorlevel% neq 0 (
        echo Falha ao criar o ambiente virtual.
        pause
        exit /b 1
    )

    echo Ativando o ambiente virtual...
    call "%VENV_DIR%\Scripts\activate.bat"

    echo Atualizando o pip...
    python -m pip install --upgrade pip >nul

    REM Verifica se o arquivo requirements.txt existe antes de instalar
    if exist "requirements.txt" (
        echo Instalando dependencias do requirements.txt...
        pip install -r requirements.txt
    ) else (
        echo AVISO: Arquivo requirements.txt nao encontrado. Nenhuma dependencia foi instalada.
    )
    
    echo.
    echo Instalacao concluida! O aplicativo sera iniciado agora.
    echo.

) else (
    REM Se a venv JÁ existir, apenas ativa.
    call "%VENV_DIR%\Scripts\activate.bat"
)

REM Roda o programa principal (executado em ambos os casos)
echo Rodando o projeto...
if exist "app\main.py" (
    python app\main.py
) else (
    echo ERRO: O arquivo 'app\main.py' nao foi encontrado.
)

pause

