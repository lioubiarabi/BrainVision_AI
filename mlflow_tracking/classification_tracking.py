
import mlflow
from pathlib import Path

# configuration MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("BrainScan")

# chemins des fichiers
# MODEL_DIR = Path("models/classification")
# REPORT_DIR = Path("report")

# recuperer le dossier racine du projet
PROJECT_DIR = Path(__file__).resolve().parent.parent
# chemins des fichiers
MODEL_DIR = PROJECT_DIR / "models" / "classification"
REPORT_DIR = PROJECT_DIR / "report"


# demarrer une experience
with mlflow.start_run(run_name="InceptionV3_FineTuning_Final"):

    # enregistrer les informations du modele
    mlflow.log_params({
        "model": "InceptionV3",
        "input_size": "299x299",
        "classes": 4
    })
    
    # enregistrer les metriques globales
    mlflow.log_metrics({
        "test_accuracy": 0.8474,
        "test_loss": 0.3743
    })

    # enregistrer les metriques par classe
    mlflow.log_metrics({
        "glioma_f1": 0.8967,
        "meningioma_f1": 0.8326,
        "notumor_f1": 0.9688,
        "pituitary_f1": 0.9375
    })


    # enregistrer les checkpoints
    for model_file in MODEL_DIR.glob("*.keras"):
        mlflow.log_artifact(str(model_file), artifact_path="models")

    # enregistrer les resultats
    for report_file in REPORT_DIR.iterdir():
        if report_file.is_file():
            mlflow.log_artifact(str(report_file), artifact_path="reports")

    print("experience InceptionV3 enregistree dans MLflow")


