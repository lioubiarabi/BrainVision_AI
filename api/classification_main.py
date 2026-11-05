
from pathlib import Path
from io import BytesIO
import base64

import numpy as np
import tensorflow as tf

from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO


# chemins des modeles
BASE_DIR = Path(__file__).resolve().parent.parent

CLASSIFICATION_MODEL_PATH = (
    BASE_DIR / "models/classification/inceptionv3_best.keras"
)

DETECTION_MODEL_PATH = (
    BASE_DIR / "models/detection/brainscan_yolo_best.pt"
)

# ordre des classes de classification
CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

IMAGE_SIZE = (299, 299)

app = FastAPI(
    title="BrainScan API",
    description="API de classification et detection des tumeurs cerebrales",
    version="1.0"
)

# chargement des modeles a la premiere utilisation
classification_model = None
detection_model = None


def get_classification_model():
    global classification_model

    if classification_model is None:
        classification_model = tf.keras.models.load_model(
            CLASSIFICATION_MODEL_PATH
        )

    return classification_model


@app.get("/")
def home():
    return {
        "message": "Bienvenue dans BrainScan API",
        "endpoints": [
            "/predict/classification",
            "/predict/detection"
        ]
    }


@app.post("/predict/classification")
async def predict_classification(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Format accepte : JPG ou PNG"
        )

    try:
        image_bytes = await file.read()

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMAGE_SIZE)

        # preprocessing identique a l'evaluation
        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        image_array = tf.keras.applications.inception_v3.preprocess_input(
            image_array
        )

        model = get_classification_model()

        probabilities = model.predict(image_array, verbose=0)[0]

        predicted_index = int(np.argmax(probabilities))

        return {
            "model": "InceptionV3",
            "predicted_class": CLASS_NAMES[predicted_index],
            "confidence": float(probabilities[predicted_index]),
            "probabilities": {
                CLASS_NAMES[i]: float(probabilities[i])
                for i in range(len(CLASS_NAMES))
            }
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
