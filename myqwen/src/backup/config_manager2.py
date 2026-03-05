import json
import os
from pathlib import Path
from utilities.streaming_audio_writer import StreamingAudioWriter
from persona_manager import PersonaManager


class ConfigManager:
    """
    Loads persona/quote data from the project JSON config and wires up
    the audio writer. All path/model/tuning values come from the resolved
    cfg namespace (CLI args → Config defaults).
    """

    def __init__(self, cfg):
        DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data/"
        config_path = DATA_DIR / "configurations/config_hamlet.json"

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        # Paths and tuning values come from cfg (CLI → Config defaults)
        self.model_path       = cfg.model_path
        self.cache_path       = cfg.cache_path
        self.script_path      = cfg.script_path
        self.output_path      = cfg.output_path
        self.seed             = cfg.seed
        self.temp             = cfg.temp
        self.streaming_rate   = cfg.streaming_rate

        # Data that only lives in the JSON config
        self.quotes = json_data.get("quotes", [])
        self.foo    = PersonaManager(DATA_DIR)

        # Audio writer
        self.writer = StreamingAudioWriter(self.output_path, sr=self.streaming_rate)