from gtts import gTTS
import os
import playsound

def text_to_speech(text):
    # Convert the text to speech using Google TTS
    tts = gTTS(text=text, lang='en', slow=False)

    # Save the generated speech to a file
    tts.save("output.mp3")

    # Play the generated audio (it will stream or play directly from the file)
    playsound.playsound("output.mp3")
    os.remove("output.mp3")

if __name__ == "__main__":
    # Example English text
    # text = "Hi Guru, thanks for mentoring Generative AI"
    text ="spatial"
    # Convert text to speech and stream audio
    text_to_speech(text)