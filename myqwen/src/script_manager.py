import re
class ScriptManager:
    def __init__(self, file_path):
        self.script_data = []
        self.load_script(file_path)
    
    def load_script(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                clean_line = line.strip()
                
                # Skip empty lines or lines without our colon separator
                if not clean_line or ":" not in clean_line:
                    continue

                # Split only on the FIRST colon
                parts = clean_line.split(":", 1)
                
                if len(parts) == 2:
                    speaker = parts[0].strip().lower().replace(" ", "_")
                    text = parts[1].strip()
                    
                    self.script_data.append({
                        "speaker": speaker, 
                        "text": text
                    })
        
    def get_script(self):
        return self.script_data


