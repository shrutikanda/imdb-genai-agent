"""
IMDB Movie Agent - Streamlit UI
Run with: streamlit run app.py
"""
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
import io

from agent import movie_agent, run_agent, TResponseInputItem

def transcribe_audio(audio_bytes):
    """Convert audio bytes to text using Google Speech Recognition"""
    recognizer = sr.Recognizer()
    try:
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def main():
    st.set_page_config(page_title="IMDB Movie Agent", layout="wide")
    st.title("🎬 IMDB Movie Agent")
    st.caption("Ask questions about movies using text or voice!")
    
    # Initialize session state
    if "convo" not in st.session_state:
        st.session_state.convo: list[TResponseInputItem] = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_audio" not in st.session_state:
        st.session_state.last_audio = None
    
    # Sidebar
    with st.sidebar:
        st.header("🎤 Voice Input")
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            icon_size="2x",
            pause_threshold=2.0,
            key="audio_recorder"
        )
        
        st.divider()
        st.header("📝 Examples")
        examples = [
            "When did The Matrix release?",
            "Top 5 movies of 2019 by meta score",
            "Top 7 comedy movies between 2010-2020",
            "Horror movies with meta > 85 and rating > 8",
            "Comedy movies about death",
            "Movies about police before 1990",
            "Al Pacino movies with rating > 8",
            "Summarize Spielberg's sci-fi movies"
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:20]}"):
                st.session_state.pending_input = ex
        
        st.divider()
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.convo = []
            st.session_state.last_audio = None
            st.rerun()
    
    # Process voice input
    if audio_bytes and audio_bytes != st.session_state.last_audio:
        st.session_state.last_audio = audio_bytes
        with st.sidebar:
            st.success("Audio recorded!")
            text = transcribe_audio(audio_bytes)
            if text:
                st.write(f"**You said:** {text}")
                st.session_state.pending_input = text
            else:
                st.warning("Could not understand audio.")
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Process pending input (from voice or buttons)
    if "pending_input" in st.session_state:
        user_input = st.session_state.pending_input
        del st.session_state.pending_input
        
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("Thinking..."):
            response, st.session_state.convo = run_agent(st.session_state.convo, user_input)
        
        with st.chat_message("assistant"):
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.rerun()
    
    # Text input
    if user_input := st.chat_input("Ask about movies..."):
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("Thinking..."):
            response, st.session_state.convo = run_agent(st.session_state.convo, user_input)
        
        with st.chat_message("assistant"):
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
