class VoiceGenerationManager:
    def __init__(self, model_container, prompt_cache_manager):
        self.engine = model_container
        self.prompt = prompt_cache_manager

    def clone_voice(self, persona, dry_run=False):
        if not persona:
            return None

        #call model to create voice characteristics and cache
        prompt = self.prompt.get_cache(
            persona["id"], 
            persona["ref_audio"], 
            persona["ref_text"])

        persona["prompt"] = prompt

        # Call Qwen to generate audio
        wav, sr = self.engine.generate(persona)
        
        return {
            "wav": wav, 
            "sr": sr, 
        }

    def generate_tasks(self):
        for key in self.personas:
            yield self.clone_voice(key)
