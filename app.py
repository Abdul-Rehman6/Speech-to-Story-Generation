import gradio as gr
import whisper
import os
from elevenlabs import ElevenLabs
import google.generativeai as genai
from tempfile import NamedTemporaryFile

# Set up APIs (replace with your actual API keys)
client = ElevenLabs(api_key='Your_api_key')
genai.configure(api_key="Your_api_key")

# Load Whisper model
model = whisper.load_model("medium")

# Gemini Model Configuration
generation_config = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "max_output_tokens": 500,
    "response_mime_type": "text/plain",
}

modell = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

def process_audio(audio_file):
    """Transcribe audio, generate a story, and return audio response."""
    
    # Use the provided file path directly
    audio_path = audio_file

    # Transcribe audio using Whisper
    result = model.transcribe(audio_path, language='en')
    transcribed_text = result["text"]

    # Generate Story using Gemini model
    try:
        chat_session = modell.start_chat(history=[])
        response = chat_session.send_message(f"Generate a simple story based on: {transcribed_text}, use common names and keep it under 200 words.")
        story = response.text
    except Exception as e:
        story = f"Error generating story: {str(e)}"

    # Convert Story to Speech using Eleven Labs
    try:
        audio_stream = client.text_to_speech.convert_as_stream(
            text=story,
            voice_id="21m00Tcm4TlvDq8ikWAM",
            model_id="eleven_multilingual_v2"
        )

        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_out:
            for chunk in audio_stream:
                if hasattr(chunk, 'content'):
                    temp_audio_out.write(chunk.content)
                elif isinstance(chunk, bytes):
                    temp_audio_out.write(chunk)
            generated_audio_path = temp_audio_out.name

    except Exception as e:
        return transcribed_text, story, None, f"Error generating audio: {str(e)}"
    
    return transcribed_text, story, generated_audio_path, None

# Gradio UI
interface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath"),
    outputs=[
        gr.Textbox(label="Transcribed Text"),
        gr.Textbox(label="Generated Story"),
        gr.Audio(label="Generated Story Audio"),
        gr.Textbox(label="Error (if any)")
    ],
    title="🎙️ Voice to Text & Story Generator",
    description="Upload an audio file to transcribe and generate a story.",
)

interface.launch(share=True,debug=True)
