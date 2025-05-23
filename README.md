# 😊 Multimodal Emotion Detection App

A lightweight yet powerful emotion detection system that interprets human emotions from **text**, **image**, and **audio** inputs. This **multimodal app** uses two separate deep learning models — one for text-based sentiment, and one for facial image emotion detection. Audio inputs are transcribed into text and then passed through the text emotion classifier (no separate audio model).  
![Emotion App_text](https://github.com/user-attachments/assets/2849ed29-3a54-48b4-b6eb-b6583df8aa66)
![Emotion App_csv](https://github.com/user-attachments/assets/f1ded3a5-8e3a-40b0-9390-64648b74b78a)
![Emotion App_audio](https://github.com/user-attachments/assets/5eb0fca0-c0b4-4c6c-9294-8374424277c9)
![Emotion App_image](https://github.com/user-attachments/assets/b64daa5c-8dcd-4527-b055-084794d09b67)
![Emotion App_camera](https://github.com/user-attachments/assets/cd26592d-7175-49fc-9a53-d23864253b9f)

---

## Features

- 🧠 **Multimodal Emotion Detection** – Supports text, image, and audio (via transcription)  
- 📷 **Image Model**: Fine-tuned `ResNet18` on the FER-2013 facial expression dataset  
- 📝 **Text Model**: Fine-tuned `MiniLM` transformer on Twitter-based emotion data  
- 🔊 **Audio Input**: Uses speech recognition to convert speech to text, then runs the text model  
- 🎯 Real-time predictions with probability-based confidence scores  
- 🚀 Deployable locally via **Streamlit**

---

## Use Cases

- Mental health screening or emotional tracking tools  
- Smart feedback or sentiment systems in e-learning and customer service  
- Multimodal sentiment analysis in social media or user experience research  
- Assistive AI for communication-impaired individuals

---

## Built With

- **Python 3.x** – Programming language
- **TensorFlow / Keras** – For training image emotion detection CNN models (Mini-Xception, ResNet18)
- **Keras ImageDataGenerator** – Image data augmentation and preprocessing
- **PyTorch** – For training text emotion detection model (MiniLM)
- **Hugging Face Transformers** – Pretrained MiniLM tokenizer and model for text classification
- **SpeechRecognition** – Audio transcription (speech-to-text)
- **OpenCV / PIL** – Image processing and camera input handling
- **Streamlit** – Interactive multi-modal app interface (text, image, audio inputs)
- **Scikit-learn** – Dataset splitting and evaluation metrics (classification reports, accuracy)

---
