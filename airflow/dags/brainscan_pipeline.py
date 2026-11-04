from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from datetime import datetime
from pathlib import Path
import json

from airflow.sdk import dag, task


DATA_DIR = Path("/opt/brainscan/data")
RESULTS_DIR = Path("/opt/brainscan/results")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dag(
    dag_id="brainscan_pipeline",
    description="pipeline de verification des datasets BrainScan",
    start_date=datetime(2026, 11, 3),
    schedule=None,
    catchup=False,
    tags=["BrainScan", "IA"],
)
def brainscan_pipeline():

    @task
    def validate_datasets():

        report = {
            "classification": {},
            "detection": {},
            "warnings": [],
        }

        # verification du dataset de classification
        classification_dir = DATA_DIR / "classification"
        classes = ["glioma", "meningioma", "notumor", "pituitary"]

        for split in ["train", "val", "test"]:
            split_dir = classification_dir / split

            if not split_dir.exists():
                raise FileNotFoundError(f"dossier manquant : {split_dir}")

            report["classification"][split] = {}

            for class_name in classes:
                class_dir = split_dir / class_name

                if not class_dir.exists():
                    raise FileNotFoundError(f"Classe manquante : {class_dir}")

                images = [
                    file for file in class_dir.iterdir()
                    if file.suffix.lower() in IMAGE_EXTENSIONS
                ]

                report["classification"][split][class_name] = len(images)

        # verification du dataset de detection
        detection_dir = DATA_DIR / "detection"
        detection_classes = [
            "Glioma",
            "Meningioma",
            "No Tumor",
            "Pituitary",
        ]

        for split in ["Train", "Val"]:
            report["detection"][split] = {}

            for class_name in detection_classes:
                class_dir = detection_dir / split / class_name
                images_dir = class_dir / "images"
                labels_dir = class_dir / "labels"

                if not images_dir.exists() or not labels_dir.exists():
                    raise FileNotFoundError(
                        f"dossier images ou labels manquant : {class_dir}"
                    )

                images = {
                    file.stem for file in images_dir.iterdir()
                    if file.suffix.lower() in IMAGE_EXTENSIONS
                }

                labels = {
                    file.stem for file in labels_dir.glob("*.txt")
                }

                missing_labels = sorted(images - labels)
                missing_images = sorted(labels - images)

                report["detection"][split][class_name] = {
                    "images": len(images),
                    "labels": len(labels),
                    "images_without_labels": missing_labels,
                    "labels_without_images": missing_images,
                }

                if missing_labels or missing_images:
                    report["warnings"].append(
                        f"{split}/{class_name}: "
                        f"{len(missing_labels)} images sans labels, "
                        f"{len(missing_images)} labels sans images"
                    )

        # sauvegarde du rapport
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        report_path = RESULTS_DIR / "dataset_validation.json"

        with open(report_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4, ensure_ascii=False)

        print(json.dumps(report, indent=4, ensure_ascii=False))
        print(f"Rapport sauvegarde : {report_path}")

        return report


    validation = validate_datasets()

    classification = DockerOperator(
        task_id="evaluate_classification",
        image="brainscan-ai:1.0",
        command="python scripts/evaluate_classification.py",
        docker_url="unix://var/run/docker.sock",
        docker_conn_id=None,
        mounts=[
            Mount(source="C:/BrainScan-v2", target="/opt/brainscan", type="bind")
        ],
        working_dir="/opt/brainscan",
        mount_tmp_dir=False,
        auto_remove="success",
    )

    detection = DockerOperator(
        task_id="evaluate_detection",
        image="brainscan-ai:1.0",
        command="python scripts/evaluate_detection.py",
        docker_url="unix://var/run/docker.sock",
        docker_conn_id=None,
        mounts=[
            Mount(source="C:/BrainScan-v2", target="/opt/brainscan", type="bind")
        ],
        working_dir="/opt/brainscan",
        mount_tmp_dir=False,
        auto_remove="success",
    )

    validation >> [classification, detection]


brainscan_pipeline()
