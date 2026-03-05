import os
from model_core import QwenModelContainer
from config_manager import ConfigLoader
from persona_manager import PromptManager
from voice_service import VoiceGenerationService
from parse_script import parse_script
import sounddevice as sd


# 1. Load your config and your script.
metadata = ConfigLoader("data/configurations/config_socially.json")

# Initialization 
core = QwenModelContainer(metadata.model_path)
personas = PromptManager(core, cache_dir=metadata.cache_path)
service = VoiceGenerationService(core, metadata, personas)

# 2. Iterate through the lines in the script as name/value pairs
script = parse_script(metadata.script_path)

for line in script:
    speaker_id = line["speaker"]

    #Lookup the Persona metadata
    persona = metadata.get_persona(speaker_id)

    if persona:
        # # Insert the dialogue into the persona
        persona["text"] = line["text"]

        # generate_audio(task)
        # tts_engine.generate(task)
        result = service.clone_voice(persona)

        print(f"\nActor: {speaker_id}")
        print(f"Dialog: {persona["text"]}.")

        # append audio to output file
        metadata.writer.write_chunk(result["wav"])
        sd.play(result["wav"], samplerate=24000) 
    else:
        print(f"✅Skipping {speaker_id}: No persona found in config.")
    
metadata.writer.close()
print(f"\n✅" + "="*30 + "\nScript Complete.")
