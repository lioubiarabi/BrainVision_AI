
from pathlib import Path
import shutil
import yaml

BASE_DIR = Path("/opt/brainscan")

SOURCE_DIR = BASE_DIR / "data/detection"
OUTPUT_DIR = SOURCE_DIR / "yolo_eval"

CLASSES = ["Glioma", "Meningioma", "Pituitary"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def main():

    images_output = OUTPUT_DIR / "images/val"
    labels_output = OUTPUT_DIR / "labels/val"

    images_output.mkdir(parents=True, exist_ok=True)
    labels_output.mkdir(parents=True, exist_ok=True)

    total_images = 0

    for class_name in CLASSES:

        source_images = SOURCE_DIR / "Val" / class_name / "images"
        source_labels = SOURCE_DIR / "Val" / class_name / "labels"

        if not source_images.exists() or not source_labels.exists():
            raise FileNotFoundError(
                f"Dossier manquant pour {class_name}"
            )

        for image_path in source_images.iterdir():

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = source_labels / f"{image_path.stem}.txt"

            if not label_path.exists():
                print(f"Label manquant : {image_path.name}")
                continue

            new_name = f"{class_name}_{image_path.name}"

            shutil.copy2(
                image_path,
                images_output / new_name
            )

            shutil.copy2(
                label_path,
                labels_output / f"{Path(new_name).stem}.txt"
            )

            total_images += 1

    # creation du fichier YAML
    data_config = {
        "path": str(OUTPUT_DIR),
        "train": "images/val",
        "val": "images/val",
        "nc": 3,
        "names": CLASSES,
    }

    with open(OUTPUT_DIR / "data.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(
            data_config,
            file,
            sort_keys=False,
            allow_unicode=True
        )

    print("\npreparation terminee !")
    print("images preparees :", total_images)
    print("dataset :", OUTPUT_DIR)
    print("configuration :", OUTPUT_DIR / "data.yaml")


if __name__ == "__main__":
    main()
