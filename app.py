import streamlit as st
from PIL import Image
from prediction import pred_class
import torch
import numpy as np

# --- Page config ---
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered",
)

# Title
st.markdown("<h1 style='text-align: center;'>🧠 Brain Tumor Classification</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Please upload a picture</h4>", unsafe_allow_html=True)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('resnet50d_checkpoint_fold0.pt', map_location=device, weights_only=False)
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Upload
uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')

    col1, col2 = st.columns([1, 1])  # แบ่งแนวนอน 2 คอลัมน์

    with col1:
        st.image(image, caption='Uploaded Image', width=200)
        if st.button("Prediction"):
            st.session_state['predict_now'] = True  # ใช้ session เพื่อให้ col2 รู้ว่าต้องแสดงผล

    if 'predict_now' in st.session_state and st.session_state['predict_now']:
        with col2:
            
            prob, pred_class_name = pred_class(model, image, class_names)
            probabilities = np.array(prob)[0]

            
            result_html = """
            <div style='
                background-color: #ffffff;
                padding: 15px 20px;
                border-radius: 8px;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.05);
                border: 4px solid black;
                width: 300px;
                text-align: left;
            '>
            """

            for i, class_name in enumerate(class_names):
                percent = float(probabilities[i]) * 100
                if class_name == pred_class_name:
                    result_html += f"<p style='color:#0000ff; font-weight:bold; font-size:18px;'>{class_name} : {percent:.2f}%</p>"
                else:
                    result_html += f"<p style='font-size:16px; color:#000;'>{class_name} : {percent:.2f}%</p>"

            result_html += "</div>"

            st.markdown(result_html, unsafe_allow_html=True)
