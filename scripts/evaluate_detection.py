
from pathlib import Path
import json

from ultralytics import YOLO

BASE_DIR = Path("/opt/brainscan")

MODEL_PATH = BASE_DIR / "models/detection/brainscan_yolo_best.pt"
DATA_YAML = BASE_DIR / "data/detection/yolo_eval/data.yaml"
RESULTS_DIR = BASE_DIR / "results/detection"


def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # charger le modele YOLO entraine
    print("Chargement du modele YOLOv8...")

    model = YOLO(str(MODEL_PATH))

    # evaluer le modele sur le dataset de validation
    print("evaluation du modele...")

    results = model.val(
        data=str(DATA_YAML),
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        plots=True,
        project=str(RESULTS_DIR),
        name="yolo_validation",
        exist_ok=True,
        verbose=True,
    )

    # recuperer les metriques
    metrics = {
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
    }

    # sauvegarder les metriques
    with open(
        RESULTS_DIR / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    # afficher les resultats
    print("\n===== RESULTATS YOLOv8 =====")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nresultats sauvegardes dans :", RESULTS_DIR)


if __name__ == "__main__":
    main()
