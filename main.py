import io
import os
import pandas as pd
import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tensorflow.keras.models import load_model as tf_load_model
from pydub import AudioSegment
from io import BytesIO

st.set_page_config(page_title="dEmotion Dashboard", layout="centered")

st.markdown("<h1 style='text-align: center;'>🧠 dEmotion</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Multimodal Emotion Detection</h3>", unsafe_allow_html=True)
st.markdown("""
<hr style="border: 0; height: 4px; background-color: #333;" />
""", unsafe_allow_html=True)


# Load models
text_model = AutoModelForSequenceClassification.from_pretrained("text_ED_model_MiniLM-L6-H384")
tokenizer = AutoTokenizer.from_pretrained("text_ED_model_MiniLM-L6-H384")
image_model = tf_load_model("image_ED_model_ResNet.h5")

# Text emotion labels
text_emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
image_emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Helper: predict emotion from text
def predict_text_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    outputs = text_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs).item()
    return text_emotion_labels[prediction]

# Helper: transcribe audio to text using microphone
def transcribe_live_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Recording... Please speak now.")
        audio = recognizer.listen(source, phrase_time_limit=5)
        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "API unavailable"

# Helper: transcribe audio to text
# def transcribe_audio_from_bytes(audio_bytes):
#     if audio_bytes is None:
#         return "❌ No audio file uploaded."
#
#     recognizer = sr.Recognizer()
#
#     try:
#         # Convert uploaded audio to WAV format using pydub
#         audio = AudioSegment.from_file(BytesIO(audio_bytes))
#         wav_bytes = BytesIO()
#         audio.export(wav_bytes, format="wav")
#         wav_bytes.seek(0)  # Reset pointer to the beginning of the file
#
#         # Use speech_recognition to transcribe the WAV file
#         with sr.AudioFile(wav_bytes) as source:
#             audio = recognizer.record(source)
#             return recognizer.recognize_google(audio)
#     except Exception as e:
#         return f"❌ Audio conversion/transcription failed: {e}"

# Helper: predict from image frame
def predict_image_emotion(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray_img[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.reshape(1, 48, 48, 1) / 255.0
        pred = image_model.predict(roi)
        emotion = image_emotion_labels[np.argmax(pred)]
        return emotion
    return "No face detected"

# Sidebar input mode
mode = st.sidebar.radio("Select Input Mode", ("Text", "Voice", "Camera"))

if mode == "Text":
    st.subheader("✍️ Enter Text")

    user_input = st.text_area("")

    if st.button("Done"):
        if user_input:
            result = predict_text_emotion(user_input)
            st.success(f"Predicted Emotion: {result}")
        else:
            st.warning("Please enter some text")

    st.markdown("---")
    st.subheader("📁 Or Upload a CSV File")

    uploaded_csv = st.file_uploader("", type=["csv"])

    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)

        # Check for 'text' column
        if "text" not in df.columns:
            st.error("CSV must have a column named 'text'")
        else:
            st.write("Predicting emotions...")
            results = []
            for i, row in df.iterrows():
                text = row["text"]
                if pd.isna(text):
                    emotion = "Invalid or Empty Text"
                else:
                    emotion = predict_text_emotion(text)
                results.append((text, emotion))

            # Show results in a table
            result_df = pd.DataFrame(results, columns=["Text", "Predicted Emotion"])
            st.dataframe(result_df)

            # Optional: Allow download
            csv_output = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results as CSV", data=csv_output, file_name="emotion_predictions.csv",
                               mime="text/csv")


elif mode == "Voice":
    # st.subheader("🎧 Upload an Audio File")
    # uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "m4a", "wav"])
    #
    # if uploaded_file is not None:
    #     audio_bytes = uploaded_file.read()
    #
    #     # Get the file extension from the filename
    #     file_extension = os.path.splitext(uploaded_file.name)[-1].replace('.', '').lower()
    #
    #     try:
    #         # Convert the uploaded audio to AudioSegment
    #         audio = AudioSegment.from_file(BytesIO(audio_bytes), format=file_extension)
    #
    #         # Export to WAV format in-memory
    #         wav_io = BytesIO()
    #         audio.export(wav_io, format="wav")
    #         wav_io.seek(0)
    #
    #         # Transcribe WAV audio
    #         transcript = transcribe_audio_from_bytes(wav_io.read())
    #
    #         if transcript:
    #             st.write(f"📝 Transcription: {transcript}")
    #         else:
    #             st.error("❌ Transcription failed.")
    #
    #     except Exception as e:
    #         st.error(f"⚠️ Error during audio conversion or transcription: {e}")
    #
    # st.markdown("---")
    st.write("### 🎙️ Use Your Microphone")

    # Custom styling for microphone record box
    st.markdown("""
        <style>
            div[data-testid="stForm"] {
                border: 2px #ccc;
                padding: 16px;
                text-align: right;
                border-radius: 10px;
                margin-top: 10px;
                background-color: #262730;
            }
            div[data-testid="stForm"] button[kind="formSubmit"] {
                background-color: #f0f2f6;
                color: black;
                padding: 10px 20px;
                font-size: 16px;
                border-radius: 5px;
                border: 1px solid #ccc;
                cursor: pointer;
            }
            div[data-testid="stForm"] button[kind="formSubmit"]:hover {
                background-color: #e4e8ec;
            }
        </style>
    """, unsafe_allow_html=True)

    # Custom microphone form
    with st.form("mic_button_form"):
        mic_trigger = st.form_submit_button("Start Microphone")

    if mic_trigger:
        transcript = transcribe_live_audio()
        st.write(f"📝 Transcription: {transcript}")
        if transcript:
            result = predict_text_emotion(transcript)
            st.success(f"🎯 Predicted Emotion: {result}")

# elif mode == "Voice":
#     uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
#     if uploaded_file:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#             tmp.write(uploaded_file.read())
#             tmp_path = tmp.name
#         transcript = transcribe_audio(tmp_path)
#         st.write(f"Transcription: {transcript}")
#         if transcript:
#             result = predict_text_emotion(transcript)
#             st.success(f"Predicted Emotion: {result}")
#
#     st.markdown("---")
#     st.write("### OR")
#     st.write("Use microphone to speak below and press the button when done.")
#
#     if st.button("Record and Analyze Emotion"):
#         transcript = transcribe_live_audio()
#         st.write(f"Transcription: {transcript}")
#         if transcript:
#             result = predict_text_emotion(transcript)
#             st.success(f"Predicted Emotion: {result}")


elif mode == "Camera":
    st.subheader("📤 Upload an Image")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        emotion = predict_image_emotion(image)
        st.success(f"Predicted Emotion: {emotion}")

    st.markdown("---")
    st.subheader("🎥 Or Use Your Camera")

    # Inject CSS styling for the cam-button
    st.markdown("""
        <style>
            div[data-testid="stForm"] {
                border: 2px #ccc;
                padding: 16px;
                text-align: right;
                border-radius: 10px;
                margin-top: 10px;
                background-color: #262730;
            }
            div[data-testid="stForm"] button[kind="formSubmit"] {
                background-color: #f0f2f9;
                color: black;
                padding: 10px 20px;
                font-size: 16px;
                border-radius: 5px;
                border: 1px solid #ccc;
                cursor: pointer;
            }
            div[data-testid="stForm"] button[kind="formSubmit"]:hover {
                background-color: #e4e8ec;
            }
        </style>
    """, unsafe_allow_html=True)

    # Custom camera form
    with st.form("cam_button_form"):
        cam_trigger = st.form_submit_button("Start Camera")

    if cam_trigger:
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            emotion = predict_image_emotion(frame)
            cv2.putText(frame, emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            stframe.image(frame, channels="BGR")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()




# elif mode == "Camera":
#     uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
#
#     if uploaded_file is not None:
#         # Read the uploaded image
#         file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#         image = cv2.imdecode(file_bytes, 1)
#
#         st.image(image, caption="Uploaded Image", use_column_width=True)
#
#         # Predict emotion
#         emotion = predict_image_emotion(image)
#         st.success(f"Predicted Emotion: {emotion}")
#
#     if st.button("Start Camera"):
#         cap = cv2.VideoCapture(0)
#         face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
#         stframe = st.empty()
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             emotion = predict_image_emotion(frame)
#             cv2.putText(frame, emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#             stframe.image(frame, channels="BGR")
#
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
#
#         cap.release()
#         cv2.destroyAllWindows()