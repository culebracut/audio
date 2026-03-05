import soundfile as sf
import os

class StreamingAudioWriter:
    def __init__(self, file_path, sr, channels=1):
        self.file_path = file_path
        self.sr = sr
        self.channels = channels
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        
        # Open in 'w' mode to initialize (wipe existing file)
        self.file = sf.SoundFile(file_path, mode='w', samplerate=sr, channels=channels)

    def write_chunk(self, audio_data):
        """Appends a new chunk of audio to the open file."""
        if audio_data is not None and len(audio_data) > 0:
            self.file.write(audio_data)

    def close(self):
        """Closes the file handle properly."""
        self.file.close()

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()
