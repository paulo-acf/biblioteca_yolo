####### PANCs --> Plantas Alimentícias Não Convencionais

# pip install ultralytics -->  Biblioteca usada para treinamento de modelos de visão computacional

# ora-pro-nobis             --> Pereskia aculeata
# peixinho                  --> Stachys byzantina
# gengibre-do-brejo         --> Hedychium coronarium
# mastruz                   --> Dysphania ambrosioides
# coracao-de-bananeira      --> Musa acuminata

from ultralytics import YOLO
# YOLO (classe):
#    • You Only Look Once
#    • Aqui, a rede neural analisa a imagem inteira em uma única passada

model = YOLO("yolo11n-cls.pt") # --> Objeto (instância da classe YOLO)
# YOLO( ) --> Cria um modelo YOLO
# "yolo11n-cls.pt" --> carrega um modelo pré-treinado.
# 11 --> versão do YOLO
# n --> Diferentes tamanhos de modelos (n, s, m, l, x --> nano, small, medium, large, extra-large)
# cls --> Classification (para classificação de imagens)
# .pt --> Arquivo de pesos do PyTorch (biblioteca usada para criar e treinar •redes neurais e •modelos de IA)

model.train( # --> Inicia o processo de treinamento do modelo
    data=".", # --> Especifica o local dos dados de treinamento (neste caso, o diretório atual)
    epochs=10, # --> Define o número de épocas (ciclos completos de treinamento) para o modelo
    # (20 é um número comum, mas 10 é usado aqui para um teste rápido)
    # No hub.ultralytics.com, pode-se treinar por tempo (e não por épocas) --> 1 hora, por exemplo
    # Vide --> https://www.youtube.com/watch?v=5BO0Il_YYAg
    imgsz=128 # --> Define o tamanho das imagens usadas para treinamento (128x128 [16.384 pixels])
    # (224x224 [50.176 pixels] é um tamanho comum, mas 128 é usado aqui para um teste rápido)
)