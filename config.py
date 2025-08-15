# config.py
import torch

# Rutas del dataset
root_train = "train/"
ann_train = "train/COCO_mul_train_annos.json"
root_val = "val/"
ann_val = "val/COCO_mul_val_annos.json"

# Parámetros del modelo
num_classes = 6  # fondo + tus clases
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Parámetros de entrenamiento
batch_size = 2
num_epochs = 10
lr = 0.005
momentum = 0.9
weight_decay = 0.0005
step_size = 3
gamma = 0.1