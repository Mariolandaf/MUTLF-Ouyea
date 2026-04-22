# Financial Sentiment Analysis API

*Actividad del curso de licenciatura Cloud Computing, 2026.*

## Descripción

Servicio REST que analiza el sentimiento de oraciones en inglés sobre temas financieros. Dado un texto, predice si el sentimiento es **positive**, **neutral** o **negative**.

Ejemplo de respuesta del endpoint `POST /predict`:

```json
{
  "prediction": "positive",
  "probabilities": {
    "positive": 0.9920,
    "neutral": 0.0055,
    "negative": 0.0025
  }
}
```

## Tecnologías

- **FastAPI** — servidor REST
- **DistilBERT** (HuggingFace Transformers) — modelo de lenguaje con tokenización BPE, agnóstico al vocabulario
- **PyTorch** — entrenamiento y ejecución del modelo
- **Kaggle** — dataset de 5,842 oraciones financieras etiquetadas ([sbhatti/financial-sentiment-analysis](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis))

## Pasos seguidos

### 1. Entorno virtual
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Entrenamiento (`train_model.py`)
- Descarga automática del dataset de Kaggle via `kagglehub`
- Tokenización con DistilBERT (BPE): nunca produce tokens desconocidos
- Fine-tuning de `distilbert-base-uncased` por 2 épocas sobre las 3 clases
- El modelo se guarda en `sentiment_model/`

### 3. Servidor (`main.py`)
- Al arrancar, carga el modelo desde disco
- Si `sentiment_model/` no existe, entrena automáticamente
- Expone `POST /predict` que recibe `{"text": "..."}` y devuelve la predicción

### 4. Uso
Levantar el servidor:
```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Probar desde `api_usage.ipynb` o con curl:
```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "The company reported record profits this quarter"}'
```

## Nota sobre el modelo entrenado

La carpeta `sentiment_model/` no está incluida en el repositorio (255MB). Al correr el servidor por primera vez se entrenará automáticamente (~22 horas en CPU, o menos con GPU).
