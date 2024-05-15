import streamlit as st
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from pydub import AudioSegment

# Streamlit interface for user input
uploaded_file = st.file_uploader("Choose a video file")

if uploaded_file is not None:
    # Save the uploaded file
    with open('uploaded_video.mp4', 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Extract audio from the video
    video = VideoFileClip('uploaded_video.mp4')
    audio = video.audio
    audio.write_audiofile('extracted_audio.wav')

    st.write("Audio has been extracted and saved as extracted_audio.wav")

    # Convert the audio to the correct format for pocketsphinx
    audio = AudioSegment.from_wav("extracted_audio.wav")
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio.export("converted_audio.wav", format="wav")

    st.write("Audio has been converted and saved as converted_audio.wav")

    # Transcribe the audio to text
    r = sr.Recognizer()
    with sr.AudioFile('converted_audio.wav') as source:
        audio_data = r.record(source)
        text = r.recognize_sphinx(audio_data)

    # Print the transcription and save it to a text file
    st.write("Transcription:", text)
    with open('transcription.txt', 'w') as f:
        f.write(text)

    st.write("Transcription has been saved to transcription.txt")
