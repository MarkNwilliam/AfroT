import streamlit as st
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import speech_recognition as sr
from googletrans import Translator

# Streamlit interface for user input
uploaded_file = st.file_uploader("Choose a video file")
language = st.selectbox('Select language for subtitles', ['English', 'Spanish', 'French'])

if uploaded_file is not None:
    # Save the uploaded file
    with open('uploaded_video.mp4', 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Extract audio from the video
    video = VideoFileClip('uploaded_video.mp4')
    audio = video.audio
    audio.write_audiofile('extracted_audio.wav')

    # Convert audio to text
    r = sr.Recognizer()
    audio_file = sr.AudioFile('extracted_audio.wav')

    text = ""  # Initialize text as an empty string

    with audio_file as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio)
    except sr.UnknownValueError:
        st.write("Speech Recognition could not understand audio")
    except sr.RequestError as e:
        st.write("Could not request results from Google Speech Recognition service; {0}".format(e))

    # Translate the text (if necessary)
    if language != 'English':
        translator = Translator()
        text = translator.translate(text, dest=language).text

    # Format the text into SubRip (.srt) format
    # This is a simplified example, you'll need to split the text and add timestamps
    subtitles = "1\n00:00:00,000 --> 00:00:10,000\n" + text

    # Create a text clip from the formatted subtitle text
    subtitle_clip = TextClip(subtitles, fontsize=24, color='white')

    # Merge the original video clip and the subtitle text clip
    final_clip = CompositeVideoClip([video, subtitle_clip.set_position(('center', 'bottom'))])

    # Save the final video with subtitles as a new file
    final_clip.write_videofile("final_video.mp4")

    st.write("Subtitled video has been saved as final_video.mp4")
