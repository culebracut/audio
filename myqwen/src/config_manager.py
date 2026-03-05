import json
import os
from pathlib import Path
import copy
from utilities.streaming_audio_writer import StreamingAudioWriter
from persona_manager import PersonaManager

class ConfigLoader:
    def __init__(self):

        DATA_DIR = Path(__file__).resolve().parent.parent.parent/"data/"
        config_path = DATA_DIR/"configurations/config_hamlet.json"

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found.")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # get the json data
        metadata = json_data.get("project_metadata", {})

        # save
        self.default_seed = metadata.get("seed", 42)
        self.default_temp = metadata.get("temp", 0.7)
        self.default_streaming_rate = metadata.get("default_streaming_rate", 24000)
        self.script_path = DATA_DIR/metadata.get("script_path", "")
        self.cache_path = DATA_DIR/metadata.get("cache_path")
        self.output_path = DATA_DIR/metadata.get("output_path", "")
        self.model_path = metadata.get("model_path")
        
        # 1. Load Quotes for lookup
        self.quotes = json_data.get('quotes', [])
        
        self.foo = PersonaManager(DATA_DIR)
        print(self.foo.personas['barnardo']['ref_audio']) 
        # Output: /home/system/workspace/qwen/data/audio/input/hindu/hindu_man.wav

        # 2. Load Personas into a dictionary keyed by 'id'
        #self.personas = {p['id']: p for p in json_data.get('personas', [])}

        # create a new WAV file for dialogue output
        file_path = self.output_path
        sr = self.default_streaming_rate
        writer = StreamingAudioWriter(file_path, sr=24000) 
        self.writer = writer

"""     def get_persona(self, persona_id):
        raw_persona = self.personas.get(persona_id)
        if not raw_persona:
            return None
        
        persona = copy.deepcopy(raw_persona)
        persona['seed'] = persona.get("seed") or 42
        persona['temp'] = persona.get("temp") or 0.7
        
        # Join instructions if list
        if isinstance(persona.get("instruct"), list):
            persona["instruct"] = " ".join([str(i).strip().rstrip('.') + '.' for i in persona["instruct"]])
        
        # Cross-reference with quotes: 
        # If the persona ID is in a quote's apply_to_personas list, override text
        matching_quote = next((q['text'] for q in self.quotes if persona_id in q.get('apply_to_personas', [])), None)
        if matching_quote:
            persona["text"] = matching_quote

        return persona """

"""     def get_all_persona_ids(self):
        return list(self.personas.keys()) """
