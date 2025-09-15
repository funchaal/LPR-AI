import os
import json
import yaml
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Configuração dos modelos
MODELS_CONFIG = {
    'detection': {
        'config_path': '../models/ocr/paddlepaddle/det/en/PP-OCRv5_mobile_det/inference.yml',  # Ajuste para seu config
        'pretrained_model': '../models/ocr/paddlepaddle/det/en/PP-OCRv5_mobile_det',  # Modelo treinado
        'export_dir': '../models/exported/det_inference',
        'onnx_output': '../models/onnx/det_model.onnx',
        'openvino_output': '../models/openvino/det_model',
        'input_shape': [1, 3, 640, 640],  # [batch, channels, height, width]
        'opset_version': 11
    },
    'recognition': {
        'config_path': '../models/ocr/paddlepaddle/rec/en/PP-OCRv5_mobile_rec/inference.yml',
        'pretrained_model': '../models/ocr/paddlepaddle/rec/en/PP-OCRv5_mobile_rec',
        'export_dir': '../models/exported/rec_inference',
        'onnx_output': '../models/onnx/rec_model.onnx',
        'openvino_output': '../models/openvino/rec_model',
        'input_shape': [1, 3, 48, 320],  # Altura fixa, largura dinâmica
        'opset_version': 11
    }
}

class PaddleToOpenVINOConverter:
    def __init__(self, paddleocr_root: str = "../venv/Lib/site-packages/PaddleOCR"):
        self.paddleocr_root = Path(paddleocr_root)
        self.ensure_dependencies()
    
    def ensure_dependencies(self):
        """Verifica se as dependências necessárias estão instaladas."""
        required_packages = ['paddle2onnx', 'onnx', 'openvino']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print("❌ Pacotes faltando:", missing_packages)
            print("📦 Instale com: pip install", " ".join(missing_packages))
            return False
        
        print("✅ Todas as dependências estão instaladas")
        return True
    
    def export_paddle_model(self, model_type: str, config: Dict) -> bool:
        """
        Exporta o modelo PaddlePaddle treinado para formato de inferência.
        """
        print(f"\n🔄 Exportando modelo {model_type}...")
        
        # Cria diretórios necessários
        export_dir = Path(config['export_dir'])
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Comando para exportar o modelo
        export_cmd = [
            sys.executable,  # python
            str(self.paddleocr_root / "tools" / "export_model.py"),
            "-c", config['config_path'],
            "-o", f"Global.pretrained_model={config['pretrained_model']}",
            f"Global.save_inference_dir={config['export_dir']}"
        ]
        
        print(f"🔧 Executando: {' '.join(export_cmd)}")
        
        try:
            result = subprocess.run(export_cmd, capture_output=True, text=True, check=True)
            print("✅ Exportação concluída com sucesso!")
            
            # Verifica se os arquivos foram criados
            pdmodel_file = export_dir / "inference.pdmodel"
            pdiparams_file = export_dir / "inference.pdiparams"
            
            if pdmodel_file.exists() and pdiparams_file.exists():
                print(f"   📄 Criados: {pdmodel_file.name}, {pdiparams_file.name}")
                return True
            else:
                print("❌ Arquivos de inferência não foram encontrados após exportação")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Erro na exportação: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            return False
    
    def convert_paddle_to_onnx(self, model_type: str, config: Dict) -> bool:
        """
        Converte modelo PaddlePaddle para ONNX.
        """
        print(f"\n🔄 Convertendo {model_type} para ONNX...")
        
        # Cria diretório de saída
        onnx_output = Path(config['onnx_output'])
        onnx_output.parent.mkdir(parents=True, exist_ok=True)
        
        # Comando paddle2onnx
        paddle2onnx_cmd = [
            "paddle2onnx",
            "--model_dir", config['export_dir'],
            "--model_filename", "inference.pdmodel",
            "--params_filename", "inference.pdiparams",
            "--opset_version", str(config['opset_version']),
            "--save_file", str(onnx_output)
        ]
        
        # Adiciona input_shape se especificado
        if 'input_shape' in config:
            shape_values = ','.join(map(str, config['input_shape']))
            input_shape_dict = '{"x":[' + shape_values + ']}'
            paddle2onnx_cmd.extend(["--input_shape_dict", input_shape_dict])
        
        print(f"🔧 Executando: {' '.join(paddle2onnx_cmd)}")
        
        try:
            result = subprocess.run(paddle2onnx_cmd, capture_output=True, text=True, check=True)
            print("✅ Conversão para ONNX concluída!")
            
            if onnx_output.exists():
                print(f"   📄 Criado: {onnx_output}")
                return True
            else:
                print("❌ Arquivo ONNX não foi criado")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Erro na conversão para ONNX: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            return False
    
    def optimize_onnx_model(self, onnx_path: str) -> str:
        """
        Otimiza o modelo ONNX antes da conversão para OpenVINO.
        """
        try:
            import onnx
            from onnx import optimizer
            
            print(f"🔧 Otimizando modelo ONNX...")
            
            # Carrega o modelo ONNX
            model = onnx.load(onnx_path)
            
            # Lista de otimizações
            passes = [
                'eliminate_deadend',
                'eliminate_identity', 
                'eliminate_nop_transpose',
                'eliminate_unused_initializer',
                'extract_constant_to_initializer',
                'fuse_add_bias_into_conv',
                'fuse_bn_into_conv',
                'fuse_consecutive_transposes',
                'fuse_matmul_add_bias_into_gemm',
                'fuse_pad_into_conv',
                'fuse_transpose_into_gemm'
            ]
            
            # Aplica otimizações
            optimized_model = optimizer.optimize(model, passes)
            
            # Salva modelo otimizado
            optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
            onnx.save(optimized_model, optimized_path)
            
            print(f"✅ Modelo ONNX otimizado salvo: {optimized_path}")
            return optimized_path
            
        except ImportError:
            print("⚠️  ONNX optimizer não disponível, pulando otimização")
            return onnx_path
        except Exception as e:
            print(f"⚠️  Erro na otimização ONNX: {e}")
            return onnx_path
    
    def convert_onnx_to_openvino(self, model_type: str, config: Dict, onnx_path: str) -> bool:
        """
        Converte modelo ONNX para OpenVINO.
        """
        print(f"\n🔄 Convertendo {model_type} para OpenVINO...")
        
        # Cria diretório de saída
        output_dir = Path(config['openvino_output'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from openvino.tools import mo
            
            # Parâmetros para conversão
            mo_args = {
                'input_model': onnx_path,
                'output_dir': str(output_dir),
                'compress_to_fp16': True,  # Compressão para FP16
                'model_name': f"{model_type}_model"
            }
            
            # Adiciona forma de entrada se especificada
            if 'input_shape' in config:
                mo_args['input_shape'] = config['input_shape']
            
            print(f"🔧 Convertendo com openvino.tools.mo...")
            
            # Executa conversão
            mo.convert_model(**mo_args)
            
            # Verifica se os arquivos foram criados
            xml_file = output_dir / f"{model_type}_model.xml"
            bin_file = output_dir / f"{model_type}_model.bin"
            
            if xml_file.exists() and bin_file.exists():
                print(f"✅ Conversão para OpenVINO concluída!")
                print(f"   📄 Criados: {xml_file.name}, {bin_file.name}")
                return True
            else:
                print("❌ Arquivos OpenVINO não foram criados")
                return False
                
        except Exception as e:
            print(f"❌ Erro na conversão para OpenVINO: {e}")
            # Fallback para mo via linha de comando
            return self._convert_with_mo_cmd(model_type, config, onnx_path)
    
    def _convert_with_mo_cmd(self, model_type: str, config: Dict, onnx_path: str) -> bool:
        """
        Fallback: usa mo via linha de comando.
        """
        print("🔄 Tentando conversão via linha de comando...")
        
        output_dir = Path(config['openvino_output'])
        
        mo_cmd = [
            "mo",
            "--input_model", onnx_path,
            "--output_dir", str(output_dir),
            "--data_type", "FP16",
            "--model_name", f"{model_type}_model"
        ]
        
        if 'input_shape' in config:
            shape_str = "[" + ",".join(map(str, config['input_shape'])) + "]"
            mo_cmd.extend(["--input_shape", shape_str])
        
        try:
            result = subprocess.run(mo_cmd, capture_output=True, text=True, check=True)
            print("✅ Conversão via mo concluída!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Erro com mo: {e}")
            return False
    
    def benchmark_model(self, xml_path: str):
        """
        Executa benchmark do modelo OpenVINO.
        """
        print(f"\n📊 Executando benchmark...")
        
        benchmark_cmd = [
            "benchmark_app",
            "-m", xml_path,
            "-d", "CPU",
            "-niter", "100",
            "-nireq", "1"
        ]
        
        try:
            result = subprocess.run(benchmark_cmd, capture_output=True, text=True)
            print("📈 Resultados do benchmark:")
            print(result.stdout)
        except Exception as e:
            print(f"⚠️  Erro no benchmark: {e}")
    
    def convert_model(self, model_type: str) -> bool:
        """
        Pipeline completo de conversão para um modelo.
        """
        if model_type not in MODELS_CONFIG:
            print(f"❌ Tipo de modelo '{model_type}' não reconhecido")
            return False
        
        config = MODELS_CONFIG[model_type]
        
        print(f"\n🚀 Iniciando conversão completa para {model_type}")
        print("=" * 60)
        
        # Etapa 1: Exportar modelo PaddlePaddle
        if not self.export_paddle_model(model_type, config):
            return False
        
        # Etapa 2: Converter para ONNX
        if not self.convert_paddle_to_onnx(model_type, config):
            return False
        
        # Etapa 3: Otimizar ONNX (opcional)
        optimized_onnx = self.optimize_onnx_model(config['onnx_output'])
        
        # Etapa 4: Converter para OpenVINO
        if not self.convert_onnx_to_openvino(model_type, config, optimized_onnx):
            return False
        
        # Etapa 5: Benchmark (opcional)
        xml_path = Path(config['openvino_output']) / f"{model_type}_model.xml"
        if xml_path.exists():
            self.benchmark_model(str(xml_path))
        
        print(f"\n🎉 Conversão de {model_type} concluída com sucesso!")
        return True

def create_inference_script():
    """
    Cria um script de exemplo para usar os modelos OpenVINO.
    """
    inference_code = '''import cv2
import numpy as np
from openvino.runtime import Core

class OpenVINOOCR:
    def __init__(self, det_model_path, rec_model_path):
        self.core = Core()
        
        # Carrega modelos
        self.det_model = self.core.read_model(det_model_path)
        self.rec_model = self.core.read_model(rec_model_path)
        
        # Compila para CPU
        self.det_compiled = self.core.compile_model(self.det_model, "CPU")
        self.rec_compiled = self.core.compile_model(self.rec_model, "CPU")
        
        print("✅ Modelos OpenVINO carregados com sucesso!")
    
    def preprocess_det(self, image):
        """Pré-processamento para detecção."""
        # Redimensiona mantendo aspect ratio
        h, w = image.shape[:2]
        target_size = 640
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Padding para 640x640
        padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Normalização
        normalized = padded.astype(np.float32) / 255.0
        normalized = (normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        # CHW e batch
        input_tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
        return input_tensor, scale
    
    def preprocess_rec(self, image):
        """Pré-processamento para reconhecimento."""
        # Redimensiona para altura fixa
        h, w = image.shape[:2]
        target_h = 48
        target_w = int(w * target_h / h)
        
        if target_w < 10:
            target_w = 10
        if target_w > 320:
            target_w = 320
            
        resized = cv2.resize(image, (target_w, target_h))
        
        # Normalização
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - 0.5) / 0.5
        
        # CHW e batch
        input_tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
        return input_tensor
    
    def detect_text(self, image):
        """Detecta regiões de texto."""
        input_tensor, scale = self.preprocess_det(image)
        
        # Inferência
        result = self.det_compiled([input_tensor])
        output = list(result.values())[0]
        
        # Pós-processamento (simplificado)
        # Você precisa implementar o pós-processamento específico do seu modelo
        return output
    
    def recognize_text(self, text_region):
        """Reconhece texto em uma região."""
        input_tensor = self.preprocess_rec(text_region)
        
        # Inferência  
        result = self.rec_compiled([input_tensor])
        output = list(result.values())[0]
        
        # Pós-processamento (simplificado)
        # Você precisa implementar decodificação CTC e dicionário
        return output

# Exemplo de uso
if __name__ == "__main__":
    # Caminhos para os modelos OpenVINO
    det_model = "../models/openvino/det_model/detection_model.xml"
    rec_model = "../models/openvino/rec_model/recognition_model.xml"
    
    # Inicializa OCR
    ocr = OpenVINOOCR(det_model, rec_model)
    
    # Carrega imagem de teste
    image = cv2.imread("test_image.jpg")
    
    # Detecta texto
    detections = ocr.detect_text(image)
    print("Detecções:", detections.shape)
    
    # Para cada detecção, reconhece o texto
    # (implementação completa depende do seu pós-processamento)
'''
    
    with open("openvino_ocr_inference.py", "w", encoding="utf-8") as f:
        f.write(inference_code)
    
    print("📄 Script de inferência criado: openvino_ocr_inference.py")

def main():
    """
    Função principal.
    """
    print("🔄 CONVERSOR PADDLEOCR → ONNX → OPENVINO")
    print("=" * 60)
    
    # Inicializa conversor
    converter = PaddleToOpenVINOConverter()
    
    # Lista de modelos para converter
    models_to_convert = ['detection', 'recognition']
    successful_conversions = 0
    
    for model_type in models_to_convert:
        if converter.convert_model(model_type):
            successful_conversions += 1
        else:
            print(f"❌ Falha na conversão de {model_type}")
    
    print(f"\n🎯 RESUMO: {successful_conversions}/{len(models_to_convert)} modelos convertidos")
    
    if successful_conversions > 0:
        print("\n📝 Próximos passos:")
        print("1. Teste os modelos OpenVINO com o script de inferência")
        print("2. Ajuste o pós-processamento conforme necessário") 
        print("3. Execute benchmarks para verificar performance")
        
        # Cria script de exemplo
        create_inference_script()

if __name__ == '__main__':
    main()