import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib

# ---------------------------------
# App Config
# ---------------------------------
st.set_page_config(
    page_title="Melanoma Detection (PH2)",
    layout="centered"
)

st.title("🧬 Melanoma Detection System")
st.markdown("""
⚠️ **Important Notice**  
This model is trained **ONLY on PH2 dermoscopic images**.  
Results are **not valid** for mobile or clinical photographs.
""")

# ---------------------------------
# Load Model
# ---------------------------------
@st.cache_resource
def load_model():
    return joblib.load("RF_Model_V2.joblib")

model = load_model()

# ---------------------------------
# Feature Extraction Function
# ---------------------------------
def extract_features(image):
    image = cv2.resize(image, (256, 256))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Lesion pixels only (non-black)
    lesion_pixels = gray[gray > 0]

    if len(lesion_pixels) < 100:
        return None

    features = {
        "Mean_Intensity": np.mean(lesion_pixels),
        "Std_Intensity": np.std(lesion_pixels),
        "Area": len(lesion_pixels),
        "R_Mean": np.mean(image[:, :, 0][gray > 0]),
        "G_Mean": np.mean(image[:, :, 1][gray > 0]),
        "B_Mean": np.mean(image[:, :, 2][gray > 0]),
    }

    return pd.DataFrame([features])

# ---------------------------------
# Image Upload
# ---------------------------------
uploaded_file = st.file_uploader(
    "Upload PH2 Dermoscopic Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    features_df = extract_features(img)

    if features_df is None:
        st.error("❌ No valid lesion detected. Please upload a proper PH2 image.")
    else:
        st.subheader("🔍 Extracted Features")
        st.dataframe(features_df)

        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0][1]

        st.subheader("🧪 Prediction Result")

        if prediction == 1:
            st.error(f"⚠️ **Melanoma Detected** (Confidence: {probability*100:.2f}%)")
        else:
            st.success(f"✅ **No Melanoma Detected** (Confidence: {(1-probability)*100:.2f}%)")
