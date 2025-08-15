# test_model.py
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
from model import get_model
import config as cfg

# Clases de tu dataset (incluye fondo)
CLASSES = ["__background__", "headlamp", "rear_bumper", "door", "hood", "front_bumper"]

# Cargar modelo entrenado
model = get_model(cfg.num_classes)
model.load_state_dict(torch.load("fasterrcnn_damage.pth", map_location=cfg.device))
model.to(cfg.device)
model.eval()

def predict_and_show(image_path, model, threshold=0.4):
    img = Image.open(image_path).convert("RGB")
    transform = torchvision.transforms.ToTensor()
    img_t = transform(img).to(cfg.device)

    with torch.no_grad():
        preds = model([img_t])

    pred = preds[0]
    boxes = pred['boxes']
    labels = pred['labels']
    scores = pred['scores']

    keep = scores >= threshold
    boxes = boxes[keep].cpu()
    labels = labels[keep].cpu()
    scores = scores[keep].cpu()

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        box = box.tolist()
        draw.rectangle(box, outline="red", width=2)
        text = f"{CLASSES[label]}: {score:.2f}"
        draw.text((box[0], box[1] - 15 if box[1] > 15 else box[1] + 5), text, fill="red", font=font)

    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    test_folder = "test/"
    for filename in os.listdir(test_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Prediciendo {filename}")
            predict_and_show(os.path.join(test_folder, filename), model)