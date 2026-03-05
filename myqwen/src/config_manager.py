import json
import os
import copy
from utilities.streaming_audio_writer import StreamingAudioWriter

class ConfigLoader:
    def __init__(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found.")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # get the json data
        metadata = json_data.get("project_metadata", {})

        # save
        self.script_path = metadata.get("script_path", "")
        self.model_path = metadata.get("model_path")
        self.cache_path = metadata.get("cache_path")
        self.output_path = metadata.get("output_path", "")
        self.default_seed = metadata.get("seed", 42)
        self.default_temp = metadata.get("temp", 0.7)
        self.default_streaming_rate = metadata.get("default_streaming_rate", 24000)
        
        # 1. Load Quotes for lookup
        self.quotes = json_data.get('quotes', [])
        
        # 2. Load Personas into a dictionary keyed by 'id'
        self.personas = {p['id']: p for p in json_data.get('personas', [])}

        # create a new WAV file for dialogue output
        file_path = self.output_path
        sr = self.default_streaming_rate
        writer = StreamingAudioWriter(file_path, sr=24000) 
        self.writer = writer

    def get_persona(self, persona_id):
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

        return persona

    def get_all_persona_ids(self):
        return list(self.personas.keys())
