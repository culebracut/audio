import json
import os
from pathlib import Path
import copy
from utilities.streaming_audio_writer import StreamingAudioWriter
from persona_manager import PersonaManager

class AudioManager:
    def __init__(self, output_path: str, streaming_rate: int):

        # DATA_DIR = Path(__file__).resolve().parent.parent.parent/"data/"
        
        # self.foo = PersonaManager(DATA_DIR)
        # print(self.foo.personas['barnardo']['ref_audio']) 

        # create a new WAV file for dialogue output
        self.file_path = output_path
        self.sr = streaming_rate
        writer = StreamingAudioWriter(self.file_path, self.sr) 
        self.writer = writer
    
    def create_audio_stream(self) -> StreamingAudioWriter:
        """Initialize a new WAV file for dialogue output."""
        writer = StreamingAudioWriter(self.file_path, self.sr)
        return writer
    
    def close_audio_stream(self):
        """Finalize the WAV file by closing the stream."""
        self.writer.write_chunk(b"")  # Finalize the audio stream
        self.writer.close()

