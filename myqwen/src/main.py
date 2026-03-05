
from model_core import QwenModelContainer
from config_manager import ConfigManager
from cache_manager import CacheManager
from voice_manager import VoiceGenerationManager
from parse_script import parse_script

# 1. Load your config and your script

metadata = ConfigManager()
personas = metadata.foo.personas

# Initialization 
model_core = QwenModelContainer(metadata.model_path)
persona_cache = CacheManager(model_core, cache_dir=metadata.cache_path)
voice_service = VoiceGenerationManager(model_core, persona_cache)

# 2. Iterate through the lines in the script as name/value pairs
script = parse_script(metadata.script_path)

for line in script:
    speaker_id = line["speaker"]

    #Lookup the Persona metadata
    persona = personas.get(speaker_id)

    if persona:
        # # Insert the dialogue into the persona
        persona["text"] = line["text"]

        # generate_audio(task)
        result = voice_service.clone_voice(persona)

        print(f"\nActor: {speaker_id}")
        print(f"Dialog: {persona["text"]}.")

        # append audio to output file
        metadata.writer.write_chunk(result["wav"])
    else:
        print(f"✅Skipping {speaker_id}: No persona found in config.")
    
metadata.writer.close()
print(f"\n✅" + "="*30 + "\nScript Complete.")
