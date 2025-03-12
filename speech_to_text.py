import streamlit as st
import pyaudio
import wave
import os
import tempfile
import time
from faster_whisper import WhisperModel
from openai import OpenAI

def record_chunk(p, stream, chunk_length=1):
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)  # Read 1024 frames (one buffer worth of data)
        frames.append(data)  # Add the data to the frames list
    return frames

def save_audio(frames, file_path):
    # Save the accumulated frames to a WAV file
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)  # Mono audio (1 channel)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))  # 16-bit
        wf.setframerate(16000)  # 16kHz sampling rate
        wf.writeframes(b"".join(frames))  # Write collected frames to the file

def main():
    st.title("Audio Recorder")

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Streamlit button to start recording with a unique key
    if st.button("Start Recording", key="start_button"):
        # Open the audio stream
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

        st.write("Recording... Please speak.")

        # List to store all frames of the recording
        all_frames = []

        # Record for 10 seconds
        start_time = time.time()

        while True:
            frames = record_chunk(p, stream, chunk_length=1)  # Record chunk
            all_frames.extend(frames)  # Add the current chunk to the accumulated frames

            # Calculate elapsed time
            elapsed_time = time.time() - start_time

            # Show the current recording time in seconds
            st.write(f"Recording {elapsed_time} seconds...")

            # Stop recording after 10 seconds
            if elapsed_time >= 3:
                st.write("Recording stopped after 10 seconds.")
                break
        # Stop the stream when done
        stream.stop_stream()
        stream.close()

        # Save the recorded audio to a WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            file_path = temp_file.name
            save_audio(all_frames, file_path)

            # Play the recorded audio
            with open(file_path, "rb") as f:
                st.audio(f.read(), format="audio/wav")

            # Option to download the recorded file
            with open(file_path, "rb") as f:
                st.download_button(label="Download the recorded audio", data=f, file_name="recording.wav",
                                   mime="audio/wav")

# Using Faster-Whisper
            # Use a smaller model (e.g., 'base' or 'small') for CPU
            model_size = "base"  # or "small", "medium"
            # Load the Whisper model (run on CPU with INT8 for efficiency)
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            # Perform transcription on the given audio file
            segments, info = model.transcribe(file_path, beam_size=5)
            # Print detected language and the language probability
            st.write(f"Detected language '{info.language}' with probability {info.language_probability:.6f}")
            # Print the transcribed segments with timestamps
            for segment in segments:
                st.write(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
# OpenAI Whisper
            client = OpenAI(api_key="")
                # Proceed with the transcription using the converted file
            with open(file_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file)
            st.subheader("Transcription...")
            # Print the transcribed text
            transcription_response = transcription.text
            question = st.write(transcription_response)
    p.terminate()
    # Clean up the temporary file
    # os.remove(file_path)

if __name__ == "__main__":
    main()
