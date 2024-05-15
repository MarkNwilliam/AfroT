import streamlit as st
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from pydub import AudioSegment
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load translation model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang="eng_Latn", tgt_lang='yor_Latn')

# Function to translate long text
def translate_long_text(long_text, max_chunk_length, translator, max_length):
    chunks = [long_text[i:i+max_chunk_length] for i in range(0, len(long_text), max_chunk_length)]
    translated_chunks = [translator(chunk, max_length=max_length)[0] for chunk in chunks]
    translated_text = ''.join([chunk['translation_text'] for chunk in translated_chunks])
    return translated_text

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

    # Translate the transcribed text
    translated_text = translate_long_text(text, max_chunk_length=1000, translator=translator, max_length=400)

    # Print the transcription and translation
    st.write("Transcription:", text)
    st.write("Translation:", translated_text)
