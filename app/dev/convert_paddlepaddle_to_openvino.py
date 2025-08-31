import os
from openvino import convert_model, save_model
from openvino.runtime import PartialShape
from openvino.preprocess import PrePostProcessor

MODELS_TO_CONVERT = {
    'detection': {
        'local_path': '../models/ocr/paddlepaddle/det/en/en_PP-OCRv3_det_infer/inference.pdmodel',
        'output_name': 'en_PP-OCRv3_det_infer',
        'convert_args': {
            'input': PartialShape([1, 3, -1, -1]),
        }
    },
    'recognition': {
        'local_path': '../models/ocr/paddlepaddle/rec/en/en_PP-OCRv4_rec_infer/inference.pdmodel',
        'output_name': 'en_PP-OCRv4_rec_infer',
        'convert_args': {
            'input': PartialShape([1, 3, 48, -1]),
        }
    }
}

OUTPUT_DIR = "models_openvino"

# --- FIM DA CONFIGURAÇÃO ---

def main():
    """
    Executa a conversão dos modelos e anexa o pré-processamento usando a API moderna (OVC).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Iniciando a conversão dos modelos. Saída será salva em: '{OUTPUT_DIR}'")

    try:
        for model_type, config in MODELS_TO_CONVERT.items():
            input_model_path = config['local_path']
            args = config['convert_args']
            
            print(f"\n--- Processando modelo de {model_type.upper()} ---")
            print(f"Arquivo de entrada: {input_model_path}")

            if not os.path.exists(input_model_path):
                print(f"ERRO: Arquivo não encontrado em '{input_model_path}'. Pulando.")
                continue

            # --- ETAPA 1: Converter a topologia do modelo ---
            print("Convertendo a topologia do modelo...")
            ov_model = convert_model(input_model_path, **args)

            # --- ETAPA 2: Anexar o pré-processamento (se necessário) ---
            # O pré-processamento de média/escala é específico do modelo de detecção.
            if model_type == 'detection':
                print("Anexando a lógica de pré-processamento (média/escala)...")
                
                # 1. Cria um objeto PrePostProcessor para o modelo
                ppp = PrePostProcessor(ov_model)
                
                # 2. Define os passos de pré-processamento para a primeira entrada do modelo
                ppp.input(0).tensor().set_layout('NCHW')
                ppp.input(0).preprocess().mean([123.675, 116.28, 103.53])
                ppp.input(0).preprocess().scale([58.395, 57.12, 57.375])
                
                # 3. Constrói e atualiza o modelo com o pré-processamento embutido
                ov_model = ppp.build()
            
            # --- ETAPA 3: Salvar o modelo final ---
            output_model_path = os.path.join(OUTPUT_DIR, config['output_name'] + '.xml')
            save_model(ov_model, output_model_path, compress_to_fp16=True)
            
            print(f"Modelo de {model_type} finalizado com sucesso!")
            print(f"Salvo em: {output_model_path} (e .bin)")

        print("\nTodos os modelos foram processados.")
        print("Lembre-se de copiar o arquivo de dicionário de caracteres para a pasta de saída.")

    except Exception as e:
        import traceback
        print(f"\nOcorreu um erro durante a conversão: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()