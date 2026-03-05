import json
from pathlib import Path

class PersonaManager:
    def __init__(self, persona_path: str, data_root: str = None) -> None:
        # 1. Anchor to Project Root (assuming script is in Pipeline/)
        #self.PROJECT_ROOT = Path(__file__).resolve().parent.parent
        #self.DATA_DIR = self.PROJECT_ROOT / "data"
        persona_file = Path(persona_path)
             
        with open(persona_file, 'r') as f:
            json_data = json.load(f)

        # 3. Standard Dictionary Comprehension with Path Resolution
        self.personas = {
            p['id']: {
                **p,
                # Resolve relative to DATA_DIR and convert to string
                "ref_audio": str(data_root / p["ref_audio"].lstrip("/"))
            }
            for p in json_data.get('personas', [])
        }