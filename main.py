# main.py
from dataset import CocoDamageDataset, get_transform
from model import get_model
from train_utils import train_one_epoch
import torch
import config as cfg

# Datasets
dataset_train = CocoDamageDataset(cfg.root_train, cfg.ann_train, get_transform(train=True))
dataset_val = CocoDamageDataset(cfg.root_val, cfg.ann_val, get_transform(train=False))

# DataLoaders
data_loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=cfg.batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
)
data_loader_val = torch.utils.data.DataLoader(
    dataset_val, batch_size=cfg.batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x))
)

# Modelo
model = get_model(cfg.num_classes)
model.to(cfg.device)

# Optimizador y scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

# Entrenamiento
for epoch in range(cfg.num_epochs):
    loss = train_one_epoch(model, optimizer, data_loader_train, cfg.device)
    lr_scheduler.step()
    print(f"Epoch [{epoch+1}/{cfg.num_epochs}] - Loss: {loss:.4f}")

torch.save(model.state_dict(), "fasterrcnn_damage.pth")
print("Entrenamiento finalizado")