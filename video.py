import time
import streamlit as st
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from pydub import AudioSegment
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# Function to translate long text
def translate_long_text(long_text, max_chunk_length, translator, max_length):
    chunks = [long_text[i:i+max_chunk_length] for i in range(0, len(long_text), max_chunk_length)]
    translated_chunks = [translator(chunk, max_length=max_length)[0] for chunk in chunks]
    translated_text = ''.join([chunk['translation_text'] for chunk in translated_chunks])
    return translated_text

# Load documents
documents = SimpleDirectoryReader("data").load_data()

# Set models
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Ollama(model="phi3", request_timeout=360.0)

# Create index and query engine
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

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
        st.session_state.translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang="eng_Latn", tgt_lang=tgt_lang)

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
            st.session_state.text = r.recognize_sphinx(audio_data)
        progress_bar.progress(75)

        # Translate the transcribed text
        status_text.text('Translating text...')
        st.session_state.translated_text = translate_long_text(st.session_state.text, max_chunk_length=1000, translator=st.session_state.translator, max_length=400)
        progress_bar.progress(100)
        status_text.text('Translation complete!')

        # Print the transcription and translation
        st.write("Transcription:", st.session_state.text)
        st.write("Translation:", st.session_state.translated_text)

# User input for query
user_query = st.text_input("Enter your query:")

if st.button('Run User Query'):
    if 'response' not in st.session_state or user_query != st.session_state.prev_query:
        # Run query
        st.session_state.response = query_engine.query(user_query)
        response_text = str(st.session_state.response)

        # Translate response from English to the selected language
        st.session_state.response_translated = translate_long_text(response_text, max_chunk_length=1000, translator=st.session_state.translator, max_length=400)

        # Store the current query to check against next time
        st.session_state.prev_query = user_query

    # Display the response
    st.write("Response:", st.session_state.response_translated)
