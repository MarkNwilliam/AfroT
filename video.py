import streamlit as st
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from pydub import AudioSegment
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

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

    # Language selection
    if st.checkbox('Translate to Yoruba'):
        tgt_lang = 'yor_Latn'
    elif st.checkbox('Translate to Fon'):
        tgt_lang = 'fon_Latn'
    else:
        st.write("Please select a language for translation.")
        st.stop()

    if st.button('Start Translation'):
        # Load translation model and tokenizer for the selected language
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang="eng_Latn", tgt_lang=tgt_lang)

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Extract audio from the video
        status_text.text('Extracting audio from video...')
        video = VideoFileClip('uploaded_video.mp4')
        audio = video.audio
        audio.write_audiofile('extracted_audio.wav')
        progress_bar.progress(25)

        # Convert the audio to the correct format for pocketsphinx
        status_text.text('Converting audio...')
        audio = AudioSegment.from_wav("extracted_audio.wav")
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        audio.export("converted_audio.wav", format="wav")
        progress_bar.progress(50)

        # Transcribe the audio to text
        status_text.text('Transcribing audio...')
        r = sr.Recognizer()
        with sr.AudioFile('converted_audio.wav') as source:
            audio_data = r.record(source)
            text = r.recognize_sphinx(audio_data)
        progress_bar.progress(75)

        # Translate the transcribed text
        status_text.text('Translating text...')
        translated_text = translate_long_text(text, max_chunk_length=1000, translator=translator, max_length=400)
        progress_bar.progress(100)
        status_text.text('Translation complete!')

        # Print the transcription and translation
        st.write("Transcription:", text)
        st.write("Translation:", translated_text)
