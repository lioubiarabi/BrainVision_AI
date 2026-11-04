
import mlflow
from pathlib import Path

# definir le chemin du projet
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models" / "detection"

# configurer MLflow
mlflow.set_tracking_uri(
    f"sqlite:///{(PROJECT_DIR / 'mlflow.db').as_posix()}"
)

# utiliser l'experience BrainScan
mlflow.set_experiment("BrainScan")

print("Tracking URI :", mlflow.get_tracking_uri())

# verifier les fichiers
files = [
    MODEL_DIR / "brainscan_yolo_best.pt",
    MODEL_DIR / "brainscan_yolo_best.onnx",
    MODEL_DIR / "data.yaml"
]

for file in files:
    if not file.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file}")

# creer un run YOLOv8
with mlflow.start_run(run_name="YOLOv8_Detection"):

    # enregistrer les hyperparametres
    mlflow.log_params({
        "model": "YOLOv8n",
        "epochs": 36,
        "imgsz": 640,
        "batch": 16,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "classes": 3
    })

    # enregistrer les metriques
    mlflow.log_metrics({
        "precision": 0.9196,
        "recall": 0.8875,
        "mAP50": 0.9489,
        "mAP50_95": 0.6251
    })

    # enregistrer les modeles
    mlflow.log_artifact(
        str(MODEL_DIR / "brainscan_yolo_best.pt"),
        artifact_path="models"
    )

    mlflow.log_artifact(
        str(MODEL_DIR / "brainscan_yolo_best.onnx"),
        artifact_path="models"
    )

    # enregistrer la configuration du dataset
    mlflow.log_artifact(
        str(MODEL_DIR / "data.yaml"),
        artifact_path="config"
    )

print("experience YOLOv8 enregistree dans MLflow")
