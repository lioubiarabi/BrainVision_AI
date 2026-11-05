
import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO

# configuration
st.set_page_config(
    page_title="BrainScan",
    page_icon="🧠",
    layout="wide"
)

CLASSIFICATION_URL = "http://localhost:8000/predict/classification"
DETECTION_URL = "http://localhost:8001/predict/detection"

st.title("🧠 BrainScan")
st.subheader("Diagnostic des tumeurs cérébrales par Intelligence Artificielle")

st.write(
    "Importez une image IRM pour effectuer une classification "
    "ou une détection de tumeur cérébrale."
)

# image upload
uploaded_file = st.file_uploader(
    "Importer une image IRM",
    type=["jpg", "jpeg", "png"]
)

# choose task
task = st.radio(
    "Choisir une fonctionnalité",
    ["Classification", "Détection YOLOv8"],
    horizontal=True
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Image IRM importée", width=400)

    if st.button("Lancer la prédiction", type="primary"):

        with st.spinner("Analyse de l'image en cours..."):

            uploaded_file.seek(0)

            if task == "Classification":
                url = CLASSIFICATION_URL
            else:
                url = DETECTION_URL

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            try:
                response = requests.post(
                    url,
                    files=files,
                    timeout=120
                )

                if response.status_code == 200:

                    result = response.json()

                    st.success("Analyse terminée !")

                    if task == "Classification":

                        st.subheader("Résultat de classification")

                        st.write(
                            "Classe prédite :",
                            result.get("predicted_class", "Non disponible")
                        )

                        st.write(
                            "Confiance :",
                            result.get("confidence", "Non disponible")
                        )

                        # st.json(result)
                        # Afficher les probabilites des classes
                        st.markdown("### Probabilites par classe")

                        probabilities = result.get("probabilities", {})

                        for class_name, probability in probabilities.items():

                            col1, col2 = st.columns([4, 1])

                            with col1:
                                st.write(class_name.capitalize())
                                st.progress(float(probability))

                            with col2:
                                st.write(f"{probability:.2%}")

                    else:

                        st.subheader("Résultat de détection")

                        st.metric(
                            "Nombre de détections",
                            result["number_of_detections"]
                        )

                        # for detection in result["detections"]:
                        #     st.write(
                        #         f"**{detection['class']}** — "
                        #         f"Confiance : "
                        #         f"{detection['confidence']:.2%}"
                        #     )
                        for index, detection in enumerate(result["detections"], start=1):

                            st.write("Classe :", detection["class"])

                            st.write(
                                "Confiance :",
                                f"{detection['confidence']:.2%}"
                            )

                            st.write(
                                "Bounding box :",
                                detection["bbox"]
                            )

                            st.progress(detection["confidence"])

                        annotated_bytes = base64.b64decode(
                            result["annotated_image"]
                        )

                        annotated_image = Image.open(
                            BytesIO(annotated_bytes)
                        )

                        st.image(
                            annotated_image,
                            caption="Image avec les détections YOLOv8",
                            width=600
                        )

                else:
                    st.error(f"Erreur API : {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"impossible de contacter l'API : {e}")
