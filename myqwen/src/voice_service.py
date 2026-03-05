class VoiceGenerationService:
    def __init__(self, model_container, config_manager, prompt_cache_manager):
        self.engine = model_container
        self.configs = config_manager
        self.prompt = prompt_cache_manager

    def clone_voice(self, persona, dry_run=False):
        if not persona:
            return None

        #call model to create voice characteristics and cache
        prompt = self.prompt.get_cache(persona["ref_audio"], persona["ref_text"])
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
