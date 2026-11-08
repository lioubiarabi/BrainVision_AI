# BrainScan — Diagnostic des tumeurs cérébrales par Intelligence Artificielle

## 1. Présentation du projet

BrainScan est un projet d'intelligence artificielle dédié à l'analyse des images IRM cérébrales. Il utilise des modèles de Deep Learning pour réaliser deux tâches complémentaires :

- **Classification :** identifier la catégorie d'une tumeur cérébrale à partir d'une image IRM avec InceptionV3.
- **Détection et localisation :** détecter et localiser les tumeurs cérébrales avec YOLOv8.

Le projet intègre également MLflow pour le suivi des expériences, Apache Airflow pour l'orchestration du pipeline, FastAPI pour exposer les modèles sous forme d'API REST et Streamlit pour fournir une interface utilisateur.

## 2. Objectifs

- Développer un modèle de classification des images IRM avec TensorFlow/Keras.
- Détecter et localiser les tumeurs cérébrales avec YOLOv8.
- Évaluer les performances des modèles.
- Suivre les expériences et les métriques avec MLflow.
- Automatiser les étapes de validation et d'évaluation avec Apache Airflow.
- Exposer les modèles à travers des API REST avec FastAPI.
- Développer une interface interactive avec Streamlit.

## 3. Technologies utilisées

| Technologie | Utilisation |
|---|---|
| Python | Langage principal |
| TensorFlow / Keras | Classification |
| InceptionV3 | Modèle de classification |
| YOLOv8 / Ultralytics | Détection et localisation |
| MLflow | Suivi des expériences |
| Apache Airflow | Orchestration du pipeline |
| FastAPI | Création des API REST |
| Streamlit | Interface utilisateur |
| Docker | Conteneurisation |
| OpenCV | Traitement des images |
| Scikit-learn | Évaluation des modèles |

## 4. Architecture du projet

```text
BrainScan/
│
├── api/
│   ├── classification_main.py
│   └── detection_main.py
│
├── app/
│   └── streamlit_app.py
│
├── airflow/
│   └── dags/
│       └── brainscan_pipeline.py
│
├── data/
│   ├── classification/
│   └── detection/
│
├── models/
│   ├── classification/
│   │   └── inceptionv3_best.keras
│   │
│   └── detection/
│       ├── brainscan_yolo_best.pt
│       └── brainscan_yolo_best.onnx
│
├── results/
│   ├── classification/
│   ├── detection/
│   └── dataset_validation.json
│
├── scripts/
│   ├── evaluate_classification.py
│   └── evaluate_detection.py
│
└── README.md
```

## 5. Modèles développés

### 5.1 Classification — InceptionV3

Le modèle InceptionV3 est développé avec TensorFlow/Keras.

Il permet de classifier une image IRM dans l'une des quatre catégories :

- Glioma
- Meningioma
- No Tumor
- Pituitary

**Prétraitement :**
- Redimensionnement des images en 299 × 299 pixels.
- Prétraitement avec `preprocess_input`.
- Utilisation des ensembles Train, Validation et Test.

**Performances obtenues sur le jeu de test :**

| Métrique | Résultat |
|---|---:|
| Accuracy | 91.08 % |
| Precision Macro | 90.99 % |
| Recall Macro | 90.91 % |
| F1-score Macro | 90.89 % |

Modèle sauvegardé : `models/classification/inceptionv3_best.keras`

### 5.2 Détection — YOLOv8

YOLOv8 est utilisé pour détecter et localiser les tumeurs dans les images IRM.

Classes détectées :
- Glioma
- Meningioma
- Pituitary

La catégorie No Tumor n'est pas une classe de détection.

**Métriques obtenues lors de la validation :**

| Métrique | Résultat |
|---|---:|
| Precision | 95.81 % |
| Recall | 89.54 % |
| mAP50 | 94.94 % |
| mAP50-95 | 63.02 % |

Modèles sauvegardés :
- `models/detection/brainscan_yolo_best.pt`
- `models/detection/brainscan_yolo_best.onnx`

## 6. Suivi des expériences avec MLflow

MLflow permet de suivre les expériences d'entraînement et d'évaluation des modèles.

Les informations suivies comprennent :
- Les métriques de performance.
- Les résultats des expériences.
- Les paramètres enregistrés.

## 7. Orchestration avec Apache Airflow

Le DAG `brainscan_pipeline` organise les tâches suivantes :

1. Validation des datasets de classification et de détection.
2. Évaluation du modèle de classification.
3. Évaluation du modèle de détection.

Les résultats de validation sont enregistrés dans `results/dataset_validation.json`.

## 8. API REST avec FastAPI

Deux API indépendantes sont utilisées pour servir les modèles.

| API | Port | Endpoint |
|---|---:|---|
| Classification | 8000 | `/predict/classification` |
| Détection | 8001 | `/predict/detection` |

Documentation interactive Swagger :

- Classification : http://localhost:8000/docs
- Détection : http://localhost:8001/docs

Les API reçoivent une image IRM et retournent les résultats au format JSON.

### Exemple de résultat — Classification

```json
{
  "model": "InceptionV3",
  "predicted_class": "pituitary",
  "confidence": 0.9897,
  "probabilities": {
    "glioma": 0.0077,
    "meningioma": 0.0024,
    "notumor": 0.0001,
    "pituitary": 0.9897
  }
}
```

### Exemple de résultat — Détection

```json
{
  "model": "YOLOv8",
  "detections": [
    {
      "class": "Pituitary",
      "confidence": 0.7707,
      "bbox": [116.7, 92.7, 139.0, 110.9]
    }
  ],
  "number_of_detections": 1,
  "annotated_image": "..."
}
```

## 9. Interface utilisateur — Streamlit

L'application Streamlit permet à l'utilisateur de :

1. Importer une image IRM.
2. Choisir entre la classification et la détection.
3. Lancer l'analyse.
4. Consulter la classe prédite et les probabilités pour la classification.
5. Consulter les détections, les scores de confiance et l'image annotée pour YOLOv8.

Interface accessible à l'adresse :

http://localhost:8501

## 10. Exécution du projet

### Prérequis

- Docker Desktop
- Docker configuré en mode Linux containers
- Image Docker `brainscan-ai:1.0`
- Projet BrainScan disponible localement

### Lancer l'API de classification

Depuis PowerShell, à la racine du projet :

```powershell
docker run -d -p 8000:8000 `
  -v "${PWD}:/opt/brainscan" `
  --name brainscan-classification-api `
  brainscan-ai:1.0 `
  uvicorn api.classification_main:app --host 0.0.0.0 --port 8000
```

### Lancer l'API de détection

```powershell
docker run -d -p 8001:8000 `
  -v "${PWD}:/opt/brainscan" `
  --name brainscan-detection-api `
  brainscan-ai:1.0 `
  uvicorn api.detection_main:app --host 0.0.0.0 --port 8000
```

### Lancer Streamlit

```powershell
docker run --rm -p 8501:8501 `
  -v "${PWD}:/opt/brainscan" `
  --name brainscan-streamlit `
  brainscan-ai:1.0 `
  streamlit run /opt/brainscan/app/streamlit_app.py --server.address 0.0.0.0
```

### Accès aux applications

| Service | URL |
|---|---|
| Streamlit | http://localhost:8501 |
| API Classification | http://localhost:8000/docs |
| API Détection | http://localhost:8001/docs |
| Airflow | http://localhost:8080 |

## 11. Résultats

Le projet fournit deux approches complémentaires d'analyse des images IRM :

- La classification identifie la catégorie de l'image.
- La détection identifie les régions où YOLOv8 détecte une tumeur et retourne leurs coordonnées.

**Important :** BrainScan est un projet académique. Ses prédictions ne constituent pas un diagnostic médical et doivent être interprétées par des professionnels de santé.

## 12. Contact & Licence

Ce projet est distribué sous licence MIT.

