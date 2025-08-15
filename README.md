# Detección de Daños en Vehículos 🚗💥 (Entrenamiento con Faster R-CNN)

Este repositorio contiene el **código para entrenar un modelo de detección de daños en vehículos** utilizando **PyTorch** y **Torchvision** con el modelo **Faster R-CNN** preentrenado.

El objetivo es detectar y clasificar automáticamente diferentes tipos de daños visibles en imágenes de automóviles, devolviendo las **dos categorías más probables** por objeto detectado.

> ⚠️ Nota: Este repositorio corresponde **únicamente a la parte de entrenamiento**.  
> La aplicación web para el despliegue (ej. Streamlit o Hugging Face Spaces) se encuentra en otro repositorio.

---

## 📌 Descripción del Proyecto
El modelo se basa en:
- **Transfer Learning** con `fasterrcnn_resnet50_fpn` de `torchvision.models.detection`.
- Dataset personalizado en formato **COCO**, cargado mediante `CocoDamageDataset`.
- Transformaciones y preprocesamiento con `torchvision.transforms`.
- Entrenamiento con utilidades personalizadas (`train_utils.py`).

---

## 📂 Estructura del Proyecto
📦 car-damage/
┣ 📂 test/ 
┣ 📂 train/ 
┗ 📂 val/ 
┣ 📜 config.py # Parámetros de configuración
┣ 📜 dataset.py # Clase CocoDamageDataset y transformaciones
┣ 📜 main.py # Donde se ejecuta todo
┣ 📜 model.py # Definición y adaptación de Faster R-CNN
┣ 📜 predict.py # Se hacen las predicciones 
┣ 📜 README.md # Documentación
┣ 📜 test.py # Script de prueba
┣ 📜 train.py # Script principal de entrenamiento

---

## 📊 Categorías Detectadas
Ejemplo de clases que puede detectar el modelo (personalizable según dataset):
- **Headlamp** (Faro delantero)
- **Rear bumper** (Parachoques trasero)
- **Door** (Puerta)
- **Hood** (Capó)
- **Front bumper** (Parachoques delantero)

---


## 🛠️ Tecnologías Utilizadas
Python 3

PyTorch

Torchvision

Pillow

Matplotlib

COCO API (pycocotools)

📜 Licencia
Este proyecto se distribuye bajo licencia MIT.

Desarrollado como prototipo para la automatización de peritajes en el sector asegurador.