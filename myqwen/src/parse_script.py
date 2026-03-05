import re

def parse_script(file_path):
    script_data = []
    
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
                
                script_data.append({
                    "speaker": speaker, 
                    "text": text
                })
                
    return script_data

