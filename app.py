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
    initial_sidebar_state="auto"
)

# Set title 
st.title('Brain Tumor Classification')
# Set Header 
st.header('Please upload a picture')


#Load Model 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load('resnet50d_checkpoint_fold0.pt', map_location=device, weights_only=False)



# Display image & Prediction 
uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    class_name = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

    if st.button('Prediction'):
        # Prediction class
        probli = pred_class(model, image, class_name)

        # Extract probability array from tuple
        probabilities = np.array(probli[0])  # ใช้เฉพาะค่าความน่าจะเป็น

        # Handle different shapes
        if probabilities.ndim == 2:  # ถ้ามี shape (1,4) ต้องดึงค่าออกมา
            probabilities = probabilities[0]

        # Get max index
        max_index = np.argmax(probabilities)

        # Display prediction results
        st.write("## Prediction Result")
        for i in range(len(class_name)):
            color = "blue" if i == max_index else None
            value = float(probabilities[i]) * 100  # Convert to percentage
            st.write(f"## <span style='color:{color}'>{class_name[i]} : {value:.2f}%</span>", unsafe_allow_html=True)