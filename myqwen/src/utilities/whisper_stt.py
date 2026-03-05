import whisper

def transcribe_whisper(audio_path):
    # Load model (options: "tiny", "base", "small", "medium", "large")
    # "base" is a good balance of speed and accuracy
    model = whisper.load_model("base")
    
    print("Transcribing with Whisper...")
    # Transcribe the audio file
    #result = model.transcribe(audio_path)
    #result = model.transcribe(audio_path, task="translate")
    result = model.transcribe(audio_path, task="transcribe")
    
    # Return the text from the result dictionary
    return result["text"]

if __name__ == "__main__":
    file_path = "/data/qwen/data/audio/input/socially_inept/jesse2.wav"
    try:
        transcription = transcribe_whisper(file_path)
        print("\n--- Transcription ---\n")
        print(transcription)
    except Exception as e:
        print(f"An error occurred: {e}")
