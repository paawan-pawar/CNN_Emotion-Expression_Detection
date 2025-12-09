# 🎭 Emotion Detection System

## 🏗️ Project Structure

```
CNN_Emotion-Expression_Detection/
├── app.py                    # Web interface (Streamlit)
├── train.py                  # Model training script
├── realtime.py               # Real-time camera detection
├── emotion_model.h5          # Pre-trained CNN model
├── haarcascade_frontalface_default.xml  # Face detection classifier
├── requirements.txt          # Python dependencies
├── data/                     # Dataset directory (not included in repo)
│   ├── train/                # Training images
│   └── test/                 # Testing images
└── README.md                 # This file
```


A deep learning-based facial emotion recognition system that detects 7 human emotions in real-time using Convolutional Neural Networks (CNN).

![Emotion Detection Demo]
![Python]
![TensorFlow]
![Streamlit]

## ✨ Features

- **Real-time Emotion Detection**: Detect emotions from images or webcam feed
- **7 Emotion Categories**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Multiple Interfaces**:
  - 📱 Web App (Streamlit)
  - 🎥 Real-time Camera (OpenCV)
  - 🤖 Model Training Script
- **High Accuracy**: Trained on FER2013 dataset with CNN architecture
- **User-Friendly**: Simple and intuitive interface

## 📊 Emotion Categories

| Emotion | Description | 
|---------|-------------|
| 😠 Angry | Expressions of anger or frustration | 
| 🤢 Disgust | Feelings of revulsion or strong disapproval | 
| 😨 Fear | Expressions of fear or anxiety | 
| 😊 Happy | Expressions of happiness or joy | 
| 😢 Sad | Expressions of sadness or sorrow | 
| 😲 Surprise | Expressions of surprise or astonishment | 
| 😐 Neutral | Neutral or no particular emotion | 

## 🚀 Quick Start
Deployement_link: https://cnnemotion-expressiondetection-qxrywykdj4zyvf2hcyfvwb.streamlit.app/

### Prerequisites
- Python 3.8 or higher
- Webcam (for real-time detection)

