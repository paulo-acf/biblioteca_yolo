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
import os # --> Módulo padrão do Python que dá acesso a pastas, arquivos etc.

BASE_DIR = os.path.dirname(__file__)
# BASE_DIR:
#    • Pega a pasta onde o arquivo .py está localizado
#    • Está em maiúscula porque é uma constante (é um valor que não deve mudar)
# __file__ --> Variável especial do Python que representa o caminho do arquivo atual
# os.path.dirname(...) --> Função que remove o nome do arquivo e deixa apenas o nome do diretório

model = YOLO("yolo11n-cls.pt") # --> Objeto (instância da classe YOLO)
# YOLO( ) --> Cria um modelo YOLO
# "yolo11n-cls.pt" --> carrega um modelo pré-treinado.
# 11 --> versão do YOLO
# n --> Diferentes tamanhos de modelos (n, s, m, l, x --> nano, small, medium, large, extra-large)
# cls --> Classification (para classificação de imagens)
# .pt --> Arquivo de pesos do PyTorch (biblioteca usada para criar e treinar •redes neurais e •modelos de IA)

results = model.train( # --> Inicia ou prepara o modelo para o treinamento
    data=BASE_DIR, # --> Especifica o local dos dados de treinamento (se fosse apenas ".", seria o local do diretório atual)
    epochs=10, # --> Define o número de épocas (ciclos completos de treinamento) para o modelo
    # (20 é um número comum, mas 10 é usado aqui para um teste rápido)
    # No hub.ultralytics.com, pode-se treinar por tempo (e não por épocas) --> 1 hora, por exemplo
    # Vide --> https://www.youtube.com/watch?v=5BO0Il_YYAg
    imgsz=128, # --> Define o tamanho das imagens usadas para treinamento (128x128 [16.384 pixels])
    # (224x224 [50.176 pixels] é um tamanho comum, mas 128 é usado aqui para um teste rápido)
    project=os.path.join(BASE_DIR, "pancs"), # Define a pasta "pancs" dentro do diretório base do projeto
    name="train-10-epocas",
    exist_ok=True, # Se a pasta do treino já existir (no caso, train-10-epocas), o código não dará erro
    plots=True # Gera gráficos automáticos do treinamento
)

save_dir = results.save_dir # Guarda o caminho da pasta onde o treino foi salvo ("Onde estão os resultados do treinamento???")

os.startfile(os.path.join(save_dir, "results.png"))
# os.startfile --> Abre automaticamente um arquivo como se alguém tivesse clicado nele
# os.path.join(...) --> Função que junta corretamente as partes de um caminho de arquivo
os.startfile(os.path.join(save_dir, "confusion_matrix.png"))
os.startfile(os.path.join(save_dir, "confusion_matrix_normalized.png"))