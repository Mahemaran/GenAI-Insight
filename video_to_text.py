import os
import openai
from moviepy.video.io.VideoFileClip import VideoFileClip
import whisper

# Load the video file
video = VideoFileClip("C:\\Users\\DELL\\Downloads\\Goku_speech.mp4")
# Extract audio from the video
audio = video.audio
# Save the extracted audio as a WAV file
audio.write_audiofile(("output_audio.wav"))

# Audio to text
model = whisper.load_model("base")
result = model.transcribe("output_audio.wav")
transcript = result["text"]
print(transcript)
os.remove("output_audio.wav")

#Text summarization
prompt = f"Summarize the following text:\n\n{transcript}"
openai.api_key = ""
response = openai.chat.completions.create(
                model="gpt-3.5-turbo",  # Specify the model (e.g., 'gpt-3.5-turbo', 'gpt-4')
                messages=[
                    {"role": "system", "content": "Summarize the following text"},
                    {"role": "user", "content": prompt}  # User input text for summarization
                ],
                max_tokens=50,  # Control the length of the summary
                temperature=0.5,  # Adjust creativity and randomness
            )
generated_text = response.choices[0].message.content.strip()
print(generated_text)