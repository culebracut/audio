import re
from pathlib import Path

def format_script_to_single_line(input_path):
    input_file = Path(input_path)
    output_file = input_file.with_name(f"{input_file.stem}_oneline.txt")

    raw_text = input_file.read_text(encoding='utf-8')

    # UPDATED REGEX LOGIC:
    # 1. Matches Name followed by a colon: ^([A-Za-z\s]+):
    # 2. Matches the newline and any leading whitespace: \s*\n\s*
    # 3. Captures everything until the next newline as the text: (.*)
    pattern = re.compile(r'^([A-Za-z\s]+):\s*\n\s*(.*)', re.MULTILINE)
    
    # Replace with "Name: Text" on one line
    formatted = pattern.sub(r'\1: \2', raw_text)

    # Clean up: Remove empty lines and extra whitespace
    lines = [line.strip() for line in formatted.splitlines() if line.strip()]
    output_file.write_text("\n".join(lines), encoding='utf-8')
    
    print(f"✅ Formatted file created: {output_file.name}")
    return output_file


# --- Execution ---
#input_file_path = "/data/Qwen3-TTS/workspace/data/whos_on_first.txt"
#input_file_path = "/data/Qwen3-TTS/workspace/data/importance_of_being_earnest.txt"
input_file_path = "/data/Qwen3-TTS/workspace/data/hamlet.txt"
format_script_to_single_line(input_file_path)
