 #Import
from transformers import pipeline
import openai
import streamlit as st
from openai import OpenAI
from pydub import AudioSegment
import whisper
import os
from textstat.textstat import textstat

# Title for the web app
st.title("Audio to Text Transcription App")
# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "ogg", "mp4"])
# Set FFmpeg path
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg"
# choose type
model = st.sidebar.selectbox(" select type", ["Whisper API", "Whisper"])

if "Whisper API" in model and uploaded_file:
    api_key = st.sidebar.text_input("Enter API Key", key="chatbot_api_key", type="password")
    if not api_key:
        st.sidebar.error("Please enter your API key.")
    else:
        client = OpenAI(api_key=api_key)
        # Save the uploaded file temporarily
        with open("temp_audio_file", "wb") as f:
            f.write(uploaded_file.read())
     # Convert the audio to WAV format using pydub
        audio = AudioSegment.from_file("temp_audio_file")
        audio.export("temp_audio_file_converted.wav", format="wav")
        if st.button("Transcription"):
            # Proceed with the transcription using the converted file
            with open("temp_audio_file_converted.wav", "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file)
            st.subheader("Transcription...")
            # Print the transcribed text
            transcription_response = transcription.text
            question= st.write(transcription_response)
            score = textstat.flesch_kincaid_grade(transcription_response)
            st.write(f"Flesch-Kincaid Grade Level: {score}")
            # Clean up temporary file
            os.remove("temp_audio_file_converted.wav")

            # pipeline_type = st.sidebar.selectbox("select",["Text Summarization"])
            openai.api_key = api_key
            # if transcription_response and pipeline_type:
            #     if pipeline_type == "Text Summarization":
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",  # Specify the model (e.g., 'gpt-3.5-turbo', 'gpt-4')
                messages=[
                    {"role": "system", "content": "Summarize the following text"},
                    {"role": "user", "content": transcription_response}  # User input text for summarization
                ],
                max_tokens=100,  # Control the length of the summary
                temperature=0.5,  # Adjust creativity and randomness
            )
            generated_text = response.choices[0].message.content.strip()
                # Sentiment Analysis
                # elif pipeline_type == "Sentiment Analysis":
                #     response = openai.chat.completions.create(
                #         model="gpt-3.5-turbo",  # You can use other models like 'gpt-3.5-turbo' as well
                #         messages=[
                #             {"role": "system",
                #              "content": "Classify the sentiment of the following text as Positive, Negative, or Neutral"},
                #             {"role": "user", "content": transcription_response}  # User input text for summarization
                #         ],
                #         max_tokens=50,
                #         temperature=0.0,  # We set temperature to 0 for more deterministic output
                #     )
                #     generated_text = response.choices[0].message.content.strip()
            st.subheader("✅ Answer:")
            st.write(generated_text)

elif model=="Whisper" and uploaded_file:
    # Save the uploaded file temporarily
    with open("temp_audio_file", "wb") as f:
        f.write(uploaded_file.read())
    # Convert the audio to WAV format using pydub
    audio = AudioSegment.from_file("temp_audio_file")
    audio.export("temp_audio_file_converted.wav", format="wav")
    # Load Whisper Model
    whisper_model = st.sidebar.selectbox("Select Whisper model",['tiny', 'base', 'small', 'medium', 'large'])
    model = whisper.load_model(whisper_model)  # Use 'tiny', 'base', 'small', 'medium', or 'large'
    if st.button("Transcription"):
        # Perform transcription
        st.write("**Note: The answer may be based on the Whisper model. No API key is required for this model...** \n")
        result = model.transcribe("temp_audio_file")
        # Show   transcription
        st.subheader("Transcription...")
        st.write(result['text'])
        score = textstat.flesch_kincaid_grade(result['text'])
        st.write(f"Flesch-Kincaid Grade Level: {score}")
        # Clean up temporary file
        os.remove("temp_audio_file_converted.wav")