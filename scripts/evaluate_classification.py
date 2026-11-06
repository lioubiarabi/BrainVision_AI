
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# chemins du projet
BASE_DIR = Path("/opt/brainscan")

MODEL_PATH = BASE_DIR / "models/classification/inceptionv3_best.keras"
TEST_DIR = BASE_DIR / "data/classification/test"
RESULTS_DIR = BASE_DIR / "results/classification"

IMAGE_SIZE = (299, 299)
BATCH_SIZE = 4


def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # charger le modele
    print("Chargement du modele InceptionV3...")

    model = tf.keras.models.load_model(MODEL_PATH)

    # charger les images de test
    print("Chargement du dataset test...")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    class_names = test_ds.class_names
    image_paths = test_ds.file_paths

    print("Classes :", class_names)
    print("Nombre d'images :", len(image_paths))

    # appliquer le preprocessing InceptionV3
    preprocess = tf.keras.applications.inception_v3.preprocess_input

    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1

    test_ds = test_ds.with_options(options)

    test_ds = test_ds.map(
      lambda images, labels: (preprocess(images), labels),
      num_parallel_calls=1
    )

    # recuperer les vraies classes
    y_true = np.concatenate([
        labels.numpy()
        for _, labels in test_ds
    ])

    # effectuer les predictions
    print("execution des predictions...")

    probabilities = model.predict(test_ds)

    y_pred = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)

    # calculer les metriques
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    # sauvegarder les metriques
    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    with open(
        RESULTS_DIR / "classification_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=4)

    pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    ).to_csv(RESULTS_DIR / "confusion_matrix.csv")

    # sauvegarder les predictions
    predictions_df = pd.DataFrame({
        "image": [Path(path).name for path in image_paths],
        "true_label": [class_names[i] for i in y_true],
        "predicted_label": [class_names[i] for i in y_pred],
        "confidence": confidence,
        "correct": y_true == y_pred,
    })

    predictions_df.to_csv(
        RESULTS_DIR / "predictions.csv",
        index=False,
    )

    # afficher les resultats
    print("\n===== RESULTATS CLASSIFICATION =====")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\n===== RAPPORT PAR CLASSE =====")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
        )
    )

    print("\nresultats sauvegardes dans :", RESULTS_DIR)


if __name__ == "__main__":
    main()
