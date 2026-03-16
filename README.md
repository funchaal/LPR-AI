# Documentação Técnica do Sistema de LPR-AI

**Autor:** Rafael Funchal  
**Data:** 22 de outubro de 2025

## 1. Introdução

Este documento fornece uma visão técnica detalhada do sistema de Reconhecimento de Placas Veiculares (LPR - License Plate Recognition). O objetivo é descrever a arquitetura do software, o fluxo de processamento de dados e os componentes individuais de uma maneira que seja acessível tanto para desenvolvedores experientes quanto para aqueles com menos familiaridade em programação Python e Inteligência Artificial.

O sistema foi projetado para ser robusto, escalável e eficiente, capaz de processar múltiplos fluxos de vídeo (como câmeras de segurança) em tempo real para detectar, rastrear e reconhecer placas de veículos.

### 1.1. Visão Geral do Projeto

O projeto é uma aplicação de visão computacional que automatiza a leitura de placas de veículos a partir de fontes de vídeo em tempo real. Ele opera em um pipeline de várias etapas:

- **Captura de Vídeo:** O sistema adquire imagens de uma ou mais fontes, que podem ser arquivos de vídeo ou transmissões de câmeras ao vivo (RTSP).
- **Detecção de Placas:** Utilizando um modelo de Inteligência Artificial (YOLO), o sistema identifica e localiza as placas de veículos em cada quadro do vídeo.
- **Rastreamento (Tracking):** Para evitar reprocessamentos desnecessários da mesma placa, um algoritmo de rastreamento acompanha cada placa detectada ao longo de quadros consecutivos.
- **Reconhecimento de Caracteres (OCR):** Uma vez que uma placa é detectada e considerada estável, a imagem da placa é enviada para um segundo modelo de IA (OCR), que extrai e converte os caracteres da placa em texto.
- **Pós-processamento e Validação:** O texto extraído é limpo, corrigido e validado para garantir que corresponda a um formato de placa válido.
- **Armazenamento e Integração:** As leituras válidas são salvas em um banco de dados local, juntamente com uma imagem de captura, e podem ser enviadas para um sistema externo via API.

## 2. Arquitetura do Sistema

O sistema é modular, o que significa que cada parte principal de sua funcionalidade é separada em componentes ou arquivos específicos. Isso facilita a manutenção, o teste e a melhoria de partes individuais do software sem afetar o restante.

### 2.1. Estrutura de Arquivos

A tabela abaixo descreve os principais arquivos do projeto e suas responsabilidades:

| Arquivo | Descrição |
| --- | --- |
| `main.py` | **Ponto de Entrada:** Orquestra a aplicação, iniciando o processamento para uma ou múltiplas fontes de vídeo (em paralelo). |
| `engine.py` | **Motor de Processamento:** Contém a lógica principal do pipeline de LPR para uma única fonte de vídeo. |
| `config.py` / `.env` / `config.json` | **Configuração:** Gerencia todas as configurações, como caminhos de modelos, fontes de vídeo e parâmetros de algoritmos. |
| `detector.py` | **Detecção de Placas:** Encapsula o modelo YOLO para detectar placas nos quadros de vídeo. |
| `ocr.py` / `onnx_ocr.py` | **OCR:** Encapsula o modelo de Reconhecimento Óptico de Caracteres para ler o texto das placas. |
| `Tracking.py` | **Rastreamento:** Implementa o algoritmo para rastrear objetos (placas) entre os quadros. |
| `preprocess.py` / `postprocess.py` | **Processamento de Dados:** Contêm funções para preparar imagens para os modelos e para limpar/validar os resultados do OCR. |
| `db_manager.py` | **Gerenciamento de Banco de Dados:** Lida com todas as interações com o banco de dados SQLite para salvar as leituras. |
| `logger.py` | **Logging:** Configura o sistema de logs para registrar informações de execução, erros e depuração. |

### 2.2. Fluxo de Dados

O fluxo de dados pode ser visualizado como um pipeline sequencial que processa cada quadro de um vídeo. O diagrama abaixo ilustra as principais etapas e a interação entre os módulos:

1. O `main.py` lê as configurações e inicia um processo para cada fonte de vídeo definida em `config.json`.
2. Cada processo executa a função `process_source` em `engine.py`.
3. O `engine.py` abre a fonte de vídeo e começa a ler os quadros um por um.
4. Cada quadro é enviado para o `detector.py`, que retorna as coordenadas das placas detectadas.
5. As detecções são passadas para o `Tracking.py`, que atribui um ID único a cada placa e a rastreia.
6. Quando uma placa rastreada é considerada estável, sua imagem é recortada e enviada para o `ocr.py`.
7. O `ocr.py` retorna o texto da placa, que é então validado e corrigido pelo `postprocess.py`.
8. Se a leitura for válida, o `db_manager.py` a salva no banco de dados.

## 3. Componentes Detalhados

### 3.1. Configuração (`config.py`, `.env`)

A configuração é centralizada para facilitar a adaptação do sistema a diferentes ambientes sem alterar o código.

- **`.env`:** Um arquivo de texto simples que armazena variáveis de ambiente. É ideal para configurar caminhos de arquivos, credenciais de API e selecionar o hardware de processamento (CPU ou GPU). Por exemplo, a variável `PLATE_DETECTION_DEVICE=gpu` instrui o sistema a usar a GPU para a detecção de placas, o que é significativamente mais rápido.
- **`config.json`:** Define as fontes de entrada de vídeo. Permite configurar múltiplas câmeras, cada uma com seu endereço, credenciais e outras especificidades.
- **`config.py`:** Este script Python carrega as configurações do `.env` e do `config.json`, validando-as e agrupando-as em um objeto `AppSettings`. Este objeto é então importado por outros módulos, garantindo que toda a aplicação use as mesmas configurações de forma consistente.

### 3.2. Motor de Processamento (`engine.py`)

Este é o coração da aplicação, responsável por orquestrar todo o pipeline de processamento de vídeo para uma única fonte. A função `process_source` em `engine.py` encapsula a lógica principal, garantindo que as etapas de captura, detecção, rastreamento e reconhecimento de caracteres funcionem de forma coesa.

#### 3.2.1. Fluxo Detalhado de `process_source`

**Inicialização:**

- Configuração de um logger específico para a fonte de vídeo, permitindo rastreamento detalhado de eventos.
- Inicialização da `VideoSource` para gerenciar a captura de frames, com lógica de reconexão para streams (RTSP, HTTP) e detecção de fim de arquivo para vídeos.
- Carregamento e configuração dos modelos de IA: o modelo YOLO para detecção de placas e o modelo OCR para reconhecimento de caracteres. O dispositivo de inferência (CPU/GPU) é validado e normalizado com base nas configurações.
- Inicialização do `CapturesDatabase` para gerenciar o armazenamento das leituras de placas.
- Configuração do módulo `Tracking` (variáveis de classe) com parâmetros essenciais como o gerenciador de banco de dados, caminho para salvar capturas, formatos de leitura, regex de filtro, contagem máxima de frames sem detecção, e detalhes da API externa.

**Loop Principal de Captura e Processamento:**

- O sistema entra em um loop `while True` para processar frames continuamente.
- **Obtenção do Frame:** Um novo frame é solicitado à `VideoSource`. Se o frame for `None`, o sistema tenta reconectar (para streams) ou encerra (para arquivos de vídeo).
- **Pré-processamento do Frame:** Se polígonos de máscara forem definidos nas configurações, o frame é processado para focar a detecção em áreas específicas.
- **Gerenciamento de Timeout do Tracking:** O método `Tracking.newFrame()` é chamado para atualizar o estado global do rastreador, permitindo que ele gerencie timeouts para passagens de veículos.
- **Detecção YOLO:** O frame processado é enviado ao modelo YOLO (`yolo.predict`) para identificar placas veiculares. Os resultados incluem as coordenadas das caixas delimitadoras e a probabilidade de detecção.
- **Processamento de Detecções:** Para cada placa detectada:
    - **Estabilidade da Placa:** É verificado se a placa está estacionária por um determinado número de frames (`STATIONARY_FRAME_THRESHOLD`). Placas estacionárias são ignoradas para evitar reprocessamento desnecessário, a menos que uma nova passagem seja iniciada.
    - **Gerenciamento de Tracking:** A lógica verifica se um `current_track` (passagem atual) existe e se ele está em processo de fechamento após uma chamada de API bem-sucedida. Se sim, um novo `Tracking` é iniciado para a nova detecção.
    - **Recorte e Pós-processamento da Placa:** A região da placa (ROI) é recortada do frame e passa por um pós-processamento (`post_process_plate`) para melhorar a qualidade da imagem para o OCR.
    - **Validação de Tamanho:** O tamanho da imagem recortada da placa é verificado. Placas muito pequenas são ignoradas para evitar leituras imprecisas.
    - **Redimensionamento e OCR:** A imagem da placa é redimensionada e enviada ao modelo de OCR (`ocr.ocr`) para extrair o texto.
    - **Limpeza e Validação do Texto:** O texto retornado pelo OCR é limpo (removendo caracteres não alfanuméricos) e normalizado para maiúsculas. Ele então passa por validações adicionais (`post_process_plate_reading`) que incluem correções de caracteres (`char_corrections.json`) e verificação de formatos conhecidos.
- **Atualização do Tracking:** Se a leitura for válida, ela é adicionada à instância de `Tracking` correspondente (`current_track.addCapture`). O `Tracking` é responsável por consolidar essas leituras e decidir quando interagir com o banco de dados ou a API externa.
- **Cálculo de FPS (Opcional):** Se habilitado nas configurações, o FPS (Frames Por Segundo) é calculado e logado periodicamente para monitoramento de desempenho.

O `engine.py` atua, portanto, como o maestro, coordenando todas as operações do sistema para transformar um fluxo de vídeo bruto em dados de placas veiculares estruturados e confiáveis.

### 3.3. Detecção de Placas (`detector.py`)

Este componente é uma interface para o modelo de detecção de objetos YOLO (You Only Look Once). YOLO é uma rede neural profunda treinada para identificar a localização de objetos específicos em uma imagem.

**Como funciona:** O `detector.py` recebe um quadro de vídeo (uma imagem) e o passa para o modelo YOLO. O modelo retorna uma lista de "caixas delimitadoras" (bounding boxes), que são as coordenadas (x, y, largura, altura) de cada placa encontrada na imagem, junto com um score de confiança.

**Otimização:** O projeto inclui scripts como `yolo2onnx.py` e `yolo2tensorrt.py`, que são ferramentas para converter o modelo YOLO para formatos mais otimizados (ONNX, TensorRT). Essas otimizações permitem que o modelo rode mais rápido e consuma menos recursos, especialmente em GPUs NVIDIA.

### 3.4. Reconhecimento de Caracteres (`ocr.py`)

Após a detecção, a imagem da placa é recortada e enviada para o módulo de OCR. Este módulo utiliza um modelo pré-treinado ONNX (neste caso, baseado na arquitetura PaddleOCR, convertido para ONNX) para "ler" os caracteres.

**Etapas do OCR:**

1.  **Pré-processamento (`preprocess.py`):** A imagem da placa é redimensionada e normalizada para se adequar ao formato esperado pelo modelo de OCR. Isso melhora a precisão da leitura.
2.  **Inferência:** A imagem processada é passada para a rede neural de OCR, que gera uma sequência de caracteres.
3.  **Pós-processamento (`postprocess.py`):** O texto bruto retornado pelo OCR passa por várias etapas de limpeza:
    - **Correção de Caracteres (`char_corrections.json`):** Substitui caracteres que são frequentemente confundidos pelo modelo (ex: 'O' por '0', 'I' por '1').
    - **Validação de Formato:** Verifica se a leitura corresponde a formatos de placa conhecidos (ex: `LLLNLNN` para placas Mercosul, onde L é letra e N é número).

### 3.5. Rastreamento de Objetos (`Tracking.py`)

Processar cada detecção de placa em cada quadro seria ineficiente, pois a mesma placa seria lida dezenas de vezes enquanto atravessa o campo de visão da câmera. O rastreador resolve isso.

**Como funciona:** Ele recebe as detecções de cada quadro e as associa às detecções do quadro anterior com base na proximidade. Cada placa que persiste na cena recebe um ID único. O sistema então opera com base nesses IDs, decidindo ler a placa apenas uma vez, quando as condições de estabilidade e confiança são ideais. Isso também permite que o sistema saiba quando um veículo específico entrou e saiu da cena.

### 3.6. Rastreamento de Passagens (`Tracking.py`)

O módulo `Tracking.py` é fundamental para a eficiência e a inteligência do sistema de reconhecimento de placas. Ao contrário de um rastreador de objetos tradicional que foca na movimentação espacial de um objeto entre quadros, este módulo gerencia a "passagem" de um veículo. Ele agrupa múltiplas detecções de placa relacionadas a um mesmo evento, determina a leitura final mais precisa e orquestra o salvamento dos dados e a comunicação com APIs externas.

#### 3.6.1. Propósito e Importância

Em um cenário onde um veículo passa por uma câmera, a detecção de placas (via YOLO) pode ocorrer em dezenas ou centenas de quadros consecutivos. Sem um mecanismo de rastreamento de passagens, cada uma dessas detecções seria tratada como um evento isolado, resultando em:

- **Ineficiência:** Repetição desnecessária de operações de OCR e validação para a mesma placa.
- **Redundância de Dados:** Múltiplas entradas idênticas ou muito similares no banco de dados e em sistemas externos.
- **Dificuldade na Tomada de Decisão:** Complexidade em determinar qual das muitas leituras é a mais confiável e representativa.

O `Tracking.py` aborda esses desafios ao gerenciar o ciclo de vida de cada placa detectada, desde sua primeira aparição até sua saída do campo de visão da câmera, garantindo que cada "passagem" de veículo resulte em uma única e precisa leitura de placa.

#### 3.6.2. Estrutura da Classe `Tracking`

A classe `Tracking` é projetada para ser tanto um gerenciador global de todas as passagens ativas quanto a representação individual de cada passagem de veículo. Ela utiliza variáveis de classe para o estado global e métodos de instância para o estado de cada passagem.

**Variáveis de Classe (Estado Global)**

Estas variáveis são compartilhadas por todas as instâncias da classe `Tracking` e são configuradas uma única vez através do método `setup()`:

- `trackings`: Um dicionário que armazena todas as passagens de veículos que estão atualmente ativas.
- `db_manager`: Objeto para interagir com o banco de dados.
- `api_endpoint`: URL da API externa para validação das placas.
- `max_no_frame_count`: Número de quadros consecutivos sem detecção antes que uma passagem seja considerada encerrada (timeout).

**Variáveis de Instância (Estado Individual de cada Passagem)**

- `id`: Um identificador único para cada passagem.
- `readings`: Um dicionário que conta a frequência de cada leitura de placa obtida pelo OCR.
- `possibleReadings`: Uma lista de leituras que passaram nos filtros de validação iniciais.
- `finalReading`: A placa final escolhida após o processo de consolidação.
- `frames`: Uma lista de frames.
- `api_calls`: Contador de chamadas de API pendentes.
- `api_returned_200`: Flag que indica se a API já retornou um status 200 (sucesso) para esta passagem.

#### 3.6.3. Fluxo de Execução

1.  **Criação:** Uma nova instância de `Tracking` é criada quando uma placa é detectada pela primeira vez.
2.  **Adição de Capturas:** A cada nova detecção da mesma placa (identificada pelo rastreador de objetos), a leitura do OCR e a imagem do quadro são adicionadas à instância.
3.  **Consolidação:** O sistema periodicamente (ou ao final da passagem) analisa todas as `readings` coletadas. Ele usa a frequência para determinar as candidatas mais prováveis (`possibleReadings`).
4.  **Validação Externa (Opcional) e Assincronicidade da API:**
    - Se um `api_endpoint` estiver configurado e houver novas placas candidatas, o `Tracking.py` envia essas placas para validação através de uma chamada de API.
    - Crucialmente, esta chamada é feita de forma **assíncrona**, utilizando `threading.Thread`. Isso significa que a thread principal do `engine.py` (responsável pelo loop de detecção e processamento de frames) não é bloqueada enquanto aguarda a resposta da API. O sistema continua a processar novos frames e a detectar outras placas sem interrupção.
    - O `Tracking.py` mantém um contador (`api_calls`) para chamadas pendentes e um flag (`api_returned_200`) para indicar se uma resposta 200 (sucesso) já foi recebida.
    - **Comportamento com Resposta 200:** Se a API externa retornar um status 200 OK (indicando que a placa foi validada com sucesso), o flag `api_returned_200` é definido como `True`, a `finalReading` é atualizada com a leitura correta fornecida pela API, e o processo de fechamento da passagem é iniciado.
    - Neste ponto, o `engine.py` é instruído a parar de processar novas detecções para aquela placa específica (o `current_track`), permitindo que o sistema se concentre em novas placas e evite o reprocessamento desnecessário de uma placa já confirmada.
    - Se a API retornar um 204 No Content (placa não encontrada), o sistema continua tentando com outras leituras possíveis.
5.  **Finalização:** Uma passagem pode ser finalizada de duas maneiras:
    - **Timeout:** Se a placa não for detectada por `max_no_frame_count` quadros, o sistema assume que o veículo saiu de cena e inicia o fechamento.
    - **Confirmação da API:** Se a API confirma uma placa, a passagem pode ser encerrada imediatamente (como descrito acima).
6.  **Seleção Final e Salvamento:** Ao finalizar, se a API não tiver confirmado uma placa, o sistema escolhe a leitura mais frequente da lista de `possibleReadings` como a `finalReading`. Em seguida, a melhor imagem da passagem é selecionada, e os dados (placa final, imagem, timestamp) são salvos no banco de dados.

## 4. Conclusão

O sistema de LPR-AI é uma aplicação complexa, mas bem estruturada, que combina técnicas de processamento de vídeo, Inteligência Artificial e engenharia de software para resolver um problema do mundo real. Sua arquitetura modular e configuração centralizada o tornam adaptável e de fácil manutenção.

Para não-especialistas, a principal ideia a reter é que o sistema funciona como uma linha de montagem digital: uma fonte de vídeo entra, e o software, através de uma série de especialistas (os modelos de IA), identifica, rastreia e lê as informações de interesse, entregando um dado estruturado e útil no final.
