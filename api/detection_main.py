
from pathlib import Path
from io import BytesIO
import base64

import numpy as np
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


def get_detection_model():
    global detection_model

    if detection_model is None:
        detection_model = YOLO(str(DETECTION_MODEL_PATH))

    return detection_model


@app.get("/")
def home():
    return {
        "message": "Bienvenue dans BrainScan API",
        "endpoints": [
            "/predict/classification",
            "/predict/detection"
        ]
    }

@app.post("/predict/detection")
async def predict_detection(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Format accepte : JPG ou PNG"
        )

    try:
        image_bytes = await file.read()

        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        model = get_detection_model()

        results = model.predict(
            source=image,
            imgsz=512,
            conf=0.25,
            device="cpu",
            workers=0,
            max_det=100,
            verbose=False
        )
        result = results[0]

        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())

                detections.append({
                    "class": model.names[class_id],
                    "confidence": float(box.conf.item()),
                    "bbox": [
                        float(value)
                        for value in box.xyxy[0].tolist()
                    ]
                })

        # generer l'image avec les bounding boxes
        annotated_image = result.plot()

        # convertir BGR vers RGB
        annotated_image = annotated_image[:, :, ::-1]

        output = BytesIO()

        Image.fromarray(annotated_image).save(
            output,
            format="JPEG"
        )

        image_base64 = base64.b64encode(
            output.getvalue()
        ).decode("utf-8")

        return {
            "model": "YOLOv8",
            "detections": detections,
            "number_of_detections": len(detections),
            "annotated_image": image_base64
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
