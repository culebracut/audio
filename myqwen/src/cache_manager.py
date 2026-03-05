import os
import hashlib
import time  # Only needed here

class CacheManager:
    def __init__(self, model_container, cache_dir="persona_cache"):
        self.engine = model_container
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self._memory_cache = {}

    def _get_ts(self):
        """Helper for formatted timestamps."""
        return time.strftime("%H:%M:%S")

    # Update the method signature to accept the persona_id
    def get_cache(self, persona_id, ref_audio, ref_text):
        ts = self._get_ts()

        # 1. RAM Cache (Use persona_id as key for better clarity)
        if persona_id in self._memory_cache:
            return self._memory_cache[persona_id]

        # 2. Disk Cache - Use the persona_id directly in the filename
        # We still keep a short hash of the audio path to prevent collisions 
        # if the same ID is reused with different audio files.
        path_hash = hashlib.md5(ref_audio.encode()).hexdigest()[:4]
        filename = f"{persona_id}_{path_hash}.pt"
        cache_path = os.path.join(self.cache_dir, filename)

        if os.path.exists(cache_path):
            print(f"[{ts}] 💿 DISK LOAD: {filename}")
            prompt = self.engine.load_persona(cache_path)
        else:
            print(f"[{ts}] 🎙️ GPU ENCODING: {persona_id} ({os.path.basename(ref_audio)})")
            prompt = self.engine.create_prompt(ref_audio, ref_text)
            self.engine.save_persona(prompt, cache_path)

        self._memory_cache[persona_id] = prompt
        return prompt


