# train_model.py
import truststore
truststore.inject_into_ssl()  # Usa los certificados del sistema Windows

import kagglehub
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

MODEL_DIR = "sentiment_model"

# Las 3 clases del dataset financiero
LABEL_MAP = {"positive": 0, "neutral": 1, "negative": 2}


class SentimentDataset(Dataset):
    """Wrapper de PyTorch Dataset para los encodings del tokenizador."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def deploy_model():
    """Descarga el dataset, fine-tunea DistilBERT y guarda el modelo."""

    # --- 1. Descargar dataset de Kaggle ---
    print("Descargando dataset financiero de Kaggle...")
    path = kagglehub.dataset_download("sbhatti/financial-sentiment-analysis")

    data = pd.read_csv(path + "/data.csv")

    # Normalizar nombres de columnas a minúsculas por si acaso
    data.columns = [c.lower() for c in data.columns]

    # Limpiar filas con valores nulos
    data = data.dropna(subset=["sentence", "sentiment"])
    data["label"] = data["sentiment"].str.strip().str.lower().map(LABEL_MAP)
    data = data.dropna(subset=["label"])
    data["label"] = data["label"].astype(int)

    sentences = data["sentence"].tolist()
    labels = data["label"].tolist()

    print(f"Dataset cargado: {len(sentences)} muestras")
    print(data["sentiment"].value_counts().to_string())

    # --- 2. Tokenizar ---
    # DistilBERT usa BPE (Byte-Pair Encoding): nunca ve tokens desconocidos
    print("\nCargando tokenizador DistilBERT...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    encodings = tokenizer(
        sentences,
        truncation=True,
        padding=True,
        max_length=128,
    )
    dataset = SentimentDataset(encodings, labels)

    # --- 3. Cargar modelo base y ajustar cabeza de clasificación ---
    print("Cargando modelo DistilBERT (3 clases)...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3,
        id2label={0: "positive", 1: "neutral", 2: "negative"},
        label2id=LABEL_MAP,
    )

    # --- 4. Fine-tuning ---
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        logging_steps=50,
        save_strategy="no",
        report_to="none",  # Deshabilitar W&B u otros loggers
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("\nIniciando fine-tuning... (esto puede tardar unos minutos)")
    trainer.train()

    # --- 5. Guardar modelo y tokenizador ---
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"\nModelo guardado en: {MODEL_DIR}/")
