from ultralytics import YOLO

# Carregue seu modelo .pt treinado
model = YOLO('../models/plate/pytorch/fine_tuned_plate_detector_3.pt')

# Exporte o modelo para o formato OpenVINO
# O argumento `half=True` exporta em FP16 para melhor desempenho
model.export(format='openvino', half=True)

print("Conversão para OpenVINO concluída.")
print("Verifique o diretório 'best_openvino_model' para os arquivos .xml e .bin")

