import argparse
from pathlib import Path
from model_manager import ModelManager
from audio_manager import AudioManager
from cache_manager import CacheManager
from voice_manager import VoiceGenerationManager
from script_manager import ScriptManager
from persona_manager import PersonaManager


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class Config:
    """Default values for all pipeline settings."""
    DATA_DIR = Path(__file__).resolve().parent.parent.parent/"data/"

    DATA_PATH: str       =  DATA_DIR
    PERSONA_PATH: str     = DATA_DIR/"personas/personas.json"
    SCRIPT_PATH: str      = DATA_DIR/"scripts/hamlet.txt"
    OUTPUT_PATH: str      = DATA_DIR/"audio/output/hamlet.wav"
    CACHE_PATH: str       = DATA_DIR/"cache"
    MODEL_PATH: str       = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    STREAMING_RATE: int   = 24000
    SEED: int             = 42
    TEMP: float           = 0.8


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments, using Config constants as defaults."""

    parser = argparse.ArgumentParser(
        description="TTS — text-to-speech pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-path",      type=str,   default=Config.DATA_PATH,      help="Path to the data directory.")
    parser.add_argument("--persona-path",   type=str,   default=Config.PERSONA_PATH,   help="Path to the persona configuration file.")
    parser.add_argument("--script-path",    type=str,   default=Config.SCRIPT_PATH,    help="Path to the input script file.")
    parser.add_argument("--output-path",    type=str,   default=Config.OUTPUT_PATH,    help="Path for the rendered audio output (.wav).")
    parser.add_argument("--model-path",     type=str,   default=Config.MODEL_PATH,     help="HuggingFace model ID or local path.")
    parser.add_argument("--cache-path",     type=str,   default=Config.CACHE_PATH,     help="Directory used for model/data caching.")
    parser.add_argument("--streaming-rate", type=int,   default=Config.STREAMING_RATE, help="Audio streaming sample rate in Hz.")
    parser.add_argument("--seed",           type=int,   default=Config.SEED,           help="Random seed for reproducibility.")
    parser.add_argument("--temp",           type=float, default=Config.TEMP,           help="Sampling temperature (0.0 – 1.0).")

    return parser.parse_args()

# ---------------------------------------------------------------------------
# Script runner
# ---------------------------------------------------------------------------

class ScriptRunner:
    """Bootstraps all services from cfg and runs the TTS pipeline."""

    def __init__(self, cfg: argparse.Namespace) -> None:
        self.cfg = cfg

        # Bootstrap services from cfg
        audio_manager = AudioManager(self.cfg.output_path, self.cfg.streaming_rate)
        self.audio = audio_manager.create_audio_stream()

        persona_manager = PersonaManager(self.cfg.persona_path, self.cfg.data_path)
        self.personas = persona_manager.personas

        self.model_core    = ModelManager(self.cfg.model_path)
        self.persona_cache = CacheManager(self.model_core, cache_dir=self.cfg.cache_path)
        self.voice_service = VoiceGenerationManager(self.model_core, self.persona_cache)

    def run(self) -> None:
        """Iterate through the parsed script and generate audio for each line."""

        script_manager = ScriptManager(self.cfg.script_path)
        script =script_manager.get_script()

        for line in script:
            speaker_id = line["speaker"]
            persona    = self.personas.get(speaker_id)

            if persona:
                persona["text"] = line["text"]

                result = self.voice_service.clone_voice(persona)

                print(f"\nActor:  {speaker_id}")
                print(f"Dialog: {persona['text']}.")

                #self.metadata.writer.write_chunk(result["wav"])
                self.audio.write_chunk(result["wav"])
            else:
                print(f"⏭  Skipping {speaker_id}: no persona found in config.")

        self.audio.close()  # Finalize the audio stream

        print(f"\n✅{'=' * 30}\nScript complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg    = parse_args()
    runner = ScriptRunner(cfg)
    runner.run()