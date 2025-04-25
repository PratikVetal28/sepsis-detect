import streamlit as st
import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from PIL import Image

# ------------------------ Load Models
IMG_MODEL = load_model(r"C:\Users\vetal\Downloads\Sepsis\skin_disease.keras")
TAB_MODEL = joblib.load(r"C:\Users\vetal\Downloads\Sepsis\sepsis_full_pipeline.joblib")

IMG_SIZE = 224
SKIN_LABELS = ["cellulitis", "herpes", "normal"]
SEPSIS_THRESHOLD = 0.50
BLUE_COLOR_THRESHOLD = 0.30

# ------------------------ Helper Functions
def preprocess_for_cnn(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img[..., ::-1].astype("float32") / 255.0
    return np.expand_dims(img, 0)

def create_thermal_image(img_bgr):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)

def color_coverage(mask, lower, upper):
    hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lower, upper).mean() / 255.0

# ------------------------ Sepsis Prediction Logic
def predict(img_bgr, numeric_vec):
    probs = IMG_MODEL.predict(preprocess_for_cnn(img_bgr))[0]
    skin_class = SKIN_LABELS[int(np.argmax(probs))]
    thermal_img = create_thermal_image(img_bgr)

    # Always compute blue alert
    blue_pct = color_coverage(thermal_img, (90, 50, 50), (130, 255, 255))
    blue_alert = blue_pct > BLUE_COLOR_THRESHOLD

    if skin_class == "normal":
        return skin_class, None, False, False, thermal_img

    # If clinical data available
    if numeric_vec:
        num_proba = TAB_MODEL.predict_proba([numeric_vec])[0][1]
        cnn_sepsis_prob = probs[2]  # Index for 'normal' class
        avg_sepsis_prob = (cnn_sepsis_prob + num_proba) / 2
        sepsis_flag = avg_sepsis_prob > SEPSIS_THRESHOLD or blue_alert
        return skin_class, num_proba, blue_alert, sepsis_flag, thermal_img

    # If no clinical data
    return skin_class, None, blue_alert, blue_alert, thermal_img

# ------------------------ Streamlit App
def main():
    st.set_page_config(page_title="Sepsis Predictor App")
    st.title("AI-Based Sepsis Detection Tool")

    menu = ["Home", "Check Sepsis"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "Home":
        st.subheader("Welcome to the Sepsis Detection Tool")
        st.write("""
        This tool uses image processing and machine learning to help identify early signs of sepsis using
        microcirculation or thermal images, along with patient clinical data.
        """)

    elif choice == "Check Sepsis":
        st.subheader("Enter Patient Info")
        with st.form("info_form"):
            name = st.text_input("Name")
            age = st.number_input("Age", min_value=0, step=1)
            gender = st.radio("Gender", ["Male", "Female", "Other"])
            contact = st.text_input("Contact Number")
            patient_id = st.text_input("Patient ID")
            ward = st.text_input("Ward")
            form_submitted = st.form_submit_button("Save Info")

        st.subheader("Upload Image and Input Clinical Data")

        image_file = st.file_uploader("Upload Skin Image", type=["jpg", "jpeg", "png"])

        st.markdown("### Clinical Data (Optional)")
        clinical_inputs = {}
        keys = ["Plasma glucose", "Plasma insulin level (μU/ml)", "Diastolic Blood Pressure (mmHg)", "Skin fold thickness (mm)",
                "Serum C-reactive protein (μU/ml)", "BMI", "Serum Lactate (mmol/L)", "Age"]
        for k in keys:
            clinical_inputs[k] = st.number_input(k, step=0.1)

        numeric_vec = [v for v in clinical_inputs.values() if v != 0]
        scan_button = st.button("Run Scan")

        if scan_button and image_file:
            image = Image.open(image_file).convert('RGB')
            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            skin_class, num_proba, blue_alert, sepsis_flag, thermal_img = predict(img_bgr, numeric_vec)

            st.image(image, caption="Original Image", use_container_width=True)

            if skin_class == "normal":
                st.image(thermal_img, caption="Thermal Image", use_container_width=True, channels="BGR")
                st.success("✅ Skin Lesion: NORMAL")
                st.success("✅ Final Verdict: NORMAL — No signs of concern")
            else:
                st.image(thermal_img, caption="Thermal Image", use_container_width=True, channels="BGR")
                st.markdown(f"**Skin Lesion:** {skin_class.upper()}")

                if skin_class == "cellulitis":
                    st.warning("""Cellulitis is a bacterial skin infection that can spread rapidly.
                    If untreated, bacteria can enter the bloodstream, leading to sepsis.
                    Key signs: redness, swelling, warmth, pain, and sometimes fever or chills.""")
                elif skin_class == "herpes":
                    st.info("""Herpes typically causes cold sores or genital lesions.
                    In immunocompromised patients, it can lead to viral sepsis or secondary bacterial infection.""")

                if num_proba is not None:
                    st.markdown(f"**Tabular Sepsis Probability:** {num_proba:.2%}")

                st.markdown(f"**Final Verdict:** {'🚨 SEPSIS DETECTED' if sepsis_flag else '✅ NO SEPSIS'}")

if __name__ == '__main__':
    main()
