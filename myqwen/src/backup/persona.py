from dataclasses import dataclass, asdict
from typing import List
import json

@dataclass
class Persona:
    id: str
    language: str
    description: str
    instruct: List[str]
    ref_audio: str
    ref_text: str
    seed: int = 42
    temp: float = 0.7

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

# --- Usage Example ---

# 1. Your raw data
persona_data = {
    "id": "duke",
    "language": "English",
    "description": "John Wayne",
    "instruct":[
    "A deep, gravelly, rhythmic Western drawl.",
    "Slow, punctuated delivery.",
    "A weary, authoritative grit."
    ],
    "ref_audio": "/workspace/audio/input/JohnWayne/JW.wav",
      "ref_text": "This that the White Man calls charity is a fine thing for widows and orphans, but no warrior can accept it, for if he does, he is no longer a man and when he is no longer a man, he is nothing and better off dead."
}

# 2. Creating an object
persona_obj = Persona.from_dict(persona_data)

# 3. Accessing attributes with dot notation
print(f"ID: {persona_obj.id}")
print(f"First Instruction: {persona_obj.instruct[0]}")

# 4. Managing a list of these objects
personas_list: List[Persona] = [persona_obj]

# 5. Converting back to JSON (for saving/exporting)
json_output = json.dumps([asdict(p) for p in personas_list], indent=4)
print("\nExported JSON:")
print(json_output)
