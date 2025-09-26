# api.py
import io
import pickle
from pathlib import Path
from typing import Dict, Any
from PIL import ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms


class SimpleCNN(nn.Module):
    def __init__(self, conv1_filters: int, conv2_filters: int, conv3_filters: int,
                 dropout_rate: float, fc_hidden_size: int, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(conv1_filters)
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(conv2_filters)
        self.conv3 = nn.Conv2d(conv2_filters, conv3_filters, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(conv3_filters)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout2d = nn.Dropout2d(dropout_rate)
        self.fc_input_size = conv3_filters * 3 * 3
        self.fc1 = nn.Linear(self.fc_input_size, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout2d(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout2d(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout2d(x)
        x = x.view(-1, self.fc_input_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# =============================================================================
# 2. CARGA DEL MODELO Y PREPARACIÓN

app = FastAPI(title="API de predicción MNIST con PyTorch", description="Una API para servir un modelo CNN entrenado.")

MODEL_PATH = Path("results") / "best_mnist_cnn_model.pkl"

# modelo usado
model: SimpleCNN = None

try:
    print(f"Cargando modelo desde: {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        saved_data = pickle.load(f)
    
    # hiperparametros y el estado del modelo
    hyperparams = saved_data['hyperparams']
    model_state_dict = saved_data['model_state_dict']
    
    model = SimpleCNN(
        conv1_filters=hyperparams['conv1_filters'],
        conv2_filters=hyperparams['conv2_filters'],
        conv3_filters=hyperparams['conv3_filters'],
        dropout_rate=hyperparams['dropout_rate'],
        fc_hidden_size=hyperparams['fc_hidden_size']
    )
    
    model.load_state_dict(model_state_dict)
    
    # modelo en modo de evaluacion
    model.eval()
    
    print("¡Modelo cargado exitosamente y en modo de evaluación!")

except FileNotFoundError:
    print(f"Error: El archivo del modelo no se encontró en '{MODEL_PATH}'.")
    print("Asegúrate de haber corrido el script de entrenamiento primero.")
    model = None
except Exception as e:
    print(f"Ocurrió un error al cargar el modelo: {e}")
    model = None

# misma transformacion de imagen que en el entrenamiento
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  
])


# =============================================================================
# 3. DEFINICION DE LOS ENDPOINTS DE LA API

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API. Usa el endpoint POST /predict/ para predecir."}


@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    if not model:
        return {"error": "El modelo no está disponible. Revisa los logs del servidor."}

    contents = await image.read()
    
    pil_image = Image.open(io.BytesIO(contents)).convert('L')

    # invertir colores por gradio
    pil_image = ImageOps.invert(pil_image)

    image_tensor = image_transform(pil_image)
    
    image_tensor = image_tensor.unsqueeze(0)

    # predicción
    with torch.no_grad():
        output = model(image_tensor)
    
    # obtener la clase con mayor probabilidad
    prediction = torch.argmax(output, dim=1)
    predicted_digit = prediction.item()
    
    print(f"Imagen recibida. Predicción: {predicted_digit}")
    
    
    return {"prediction": predicted_digit}