import streamlit as st
import requests

st.title("🎙️ Audio-to-Text Transcriber")

uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("Transcribe"):
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        with st.spinner("Transcribing..."):
            response = requests.post("http://backend:8000/transcribe/", files=files)
            if response.status_code == 200:
                st.success("Transcription complete!")
                st.text_area("Result", response.json()["text"])
            else:
                st.error("Failed to transcribe audio.")
