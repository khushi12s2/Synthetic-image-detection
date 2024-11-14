# app.py
import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
from model.classifier import load_model
from utils.augmentation import augment_image

# Load the model
model = load_model()

# Streamlit App Configuration
st.set_page_config(page_title="Synthetic Image Detection", layout="wide")
st.title("Synthetic Image Detection")
st.write("An app to detect synthetic (AI-generated) images.")

# Sidebar for settings and options
st.sidebar.header("Upload and Settings")
uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
detection_threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.5)
apply_augmentation = st.sidebar.checkbox("Apply Data Augmentation", value=False)

# Classification Function
def classify_image(image, model, threshold=0.5):
    # Convert image to a tensor and normalize
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if apply_augmentation:
        image = augment_image(image)
    
    image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    # Model inference
    with torch.no_grad():
        output = model(image_tensor)
        probability = output.item()
        is_synthetic = probability > threshold
    
    return is_synthetic, probability

# Display uploaded image and process it
if uploaded_image is not None:
    # Display the original image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Classify the image
    is_synthetic, score = classify_image(image, model, detection_threshold)

    # Display classification results
    st.subheader("Detection Result")
    result_text = "Synthetic Image" if is_synthetic else "Real Image"
    st.write(f"**Result:** {result_text}")
    st.write(f"**Confidence Score:** {score:.2f}")

    # Optional: Display augmented image for reference
    if apply_augmentation:
        augmented_image = augment_image(np.array(image))
        st.image(augmented_image, caption="Augmented Image", use_column_width=True)

# Footer
st.markdown("---")
st.markdown("### About")
st.markdown("This app uses AI-based models to detect synthetic (AI-generated) images. Adjust the threshold to control sensitivity.")
