import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

st.set_page_config(page_title="Emotion Detection", page_icon="🎭")

@st.cache_resource
def load_model():
    """Load the emotion detection model"""
    return tf.keras.models.load_model("emotion_model.h5")

# Load model
model = load_model()
labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# App title
st.title("🎭 Emotion Detection App")
st.markdown("Upload an image to detect emotions")

# Sidebar
st.sidebar.header("About")
st.sidebar.info(
    "This app uses a CNN model trained on facial expressions "
    "to detect emotions from images."
)

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=['jpg', 'jpeg', 'png'],
    help="Upload a face image for emotion detection"
)

if uploaded_file is not None:
    # Display uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Process and predict
    with st.spinner('Analyzing emotion...'):
        # Convert to grayscale and resize
        image_gray = image.convert('L')
        image_resized = image_gray.resize((48, 48))
        
        # Convert to numpy array
        img_array = np.array(image_resized) / 255.0
        img_array = img_array.reshape(1, 48, 48, 1)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get results
        emotion_idx = np.argmax(predictions)
        detected_emotion = labels[emotion_idx]
        confidence = predictions[emotion_idx]
    
    # Display results
    with col2:
        st.subheader("Results")
        
        # Emotion with confidence
        st.metric(
            label="Detected Emotion",
            value=detected_emotion.capitalize(),
            delta=f"{confidence:.1%} confidence"
        )
        
        # Progress bar for confidence
        st.progress(float(confidence))
        
        # All emotions visualization
        st.subheader("All Emotion Probabilities")
        
        for i, (label, prob) in enumerate(zip(labels, predictions)):
            # Create columns for label and percentage
            col_a, col_b, col_c = st.columns([2, 5, 1])
            
            with col_a:
                st.write(f"**{label.capitalize()}**")
            
            with col_b:
                st.progress(float(prob))
            
            with col_c:
                st.write(f"{prob:.1%}")

# Camera input option
st.divider()
st.subheader("Or use your camera")

camera_image = st.camera_input("Take a picture")

if camera_image:
    image = Image.open(camera_image)
    
    # Process and predict
    with st.spinner('Analyzing emotion from camera...'):
        image_gray = image.convert('L')
        image_resized = image_gray.resize((48, 48))
        img_array = np.array(image_resized) / 255.0
        img_array = img_array.reshape(1, 48, 48, 1)
        
        predictions = model.predict(img_array, verbose=0)[0]
        emotion_idx = np.argmax(predictions)
        detected_emotion = labels[emotion_idx]
        confidence = predictions[emotion_idx]
    
    # Show results
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Captured Image', use_column_width=True)
    
    with col2:
        st.success(f"**{detected_emotion.capitalize()}** detected!")
        st.write(f"Confidence: **{confidence:.1%}**")
        
        # Quick emotions chart
        chart_data = {label: float(prob) for label, prob in zip(labels, predictions)}
        st.bar_chart(chart_data)

# Instructions
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. **Upload an image** using the file uploader
    2. **Or use your camera** to take a picture
    3. The app will analyze facial expressions
    4. Results show detected emotion with confidence percentage
    5. View all emotion probabilities in the chart
    
    **Tips for best results:**
    - Use clear, well-lit face images
    - Face should be facing the camera
    - Remove glasses if possible
    - Ensure good contrast
    """)

# Footer
st.divider()
st.caption("Built with TensorFlow & Streamlit | Emotion Detection Model")