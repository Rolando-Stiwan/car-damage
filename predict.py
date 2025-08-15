import sys
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torchvision.transforms as T

# Configurar dispositivo
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Definir clases (ajusta según tu dataset)
CLASSES = ["__background__", "headlamp", "rear_bumper", "door", "hood", "front_bumper"]

# Cargar modelo sin pesos preentrenados para evitar warnings
model = fasterrcnn_resnet50_fpn(weights=None)
num_classes = 6
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model_path = "fasterrcnn_damage.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

def predict_and_show_with_print(image_path, model, threshold=0.5, top_k=2):
    img = Image.open(image_path).convert("RGB")
    transform = T.ToTensor()
    img_t = transform(img).to(device)

    with torch.no_grad():
        preds = model([img_t])

    pred = preds[0]
    boxes = pred['boxes']
    labels = pred['labels']
    scores = pred['scores']

    # Filtrar predicciones por score
    keep = scores >= threshold
    boxes = boxes[keep].cpu()
    labels = labels[keep].cpu()
    scores = scores[keep].cpu()

    if len(boxes) == 0:
        print(f"No se detectaron partes con confianza mayor a {threshold}")
        return

    # Ordenar por score descendente y quedarse con top_k
    scores_sorted, idx_sorted = torch.sort(scores, descending=True)
    idx_topk = idx_sorted[:top_k]

    boxes = boxes[idx_topk]
    labels = labels[idx_topk]
    scores = scores[idx_topk]

    for label, score in zip(labels, scores):
        print(f"Parte detectada: {CLASSES[label]} - Confianza: {score:.2f}")

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        box = box.tolist()
        draw.rectangle(box, outline="red", width=2)
        text = f"{CLASSES[label]}: {score:.2f}"
        draw.text((box[0], box[1] - 15), text, fill="red", font=font)

    plt.figure(figsize=(10,8))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python predict.py ruta/a/tu/imagen.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    predict_and_show_with_print(image_path, model)