from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
from train_model import deploy_model, MODEL_DIR

# Si no existe el modelo entrenado, lo entrenamos primero
if not os.path.exists(MODEL_DIR):
    print("Modelo no encontrado. Iniciando entrenamiento...")
    deploy_model()

app = FastAPI()

# Cargar modelo y tokenizador desde disco
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()  # Modo inferencia (desactiva dropout, etc.)


class InputData(BaseModel):
    text: str


@app.post("/predict")
def predict(data: InputData):  # Equivalente al score.py del ejemplo
    # Tokenizar el texto de entrada
    inputs = tokenizer(
        data.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    # Inferencia sin calcular gradientes (más rápido y sin memoria extra)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    pred_id = int(torch.argmax(logits, dim=-1).item())
    pred_label = model.config.id2label[pred_id]

    return {
        "prediction": pred_label,
        "probabilities": {
            model.config.id2label[i]: round(p, 4) for i, p in enumerate(probs)
        },
    }
