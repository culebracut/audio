from qwen_tts import Qwen3TTSModel
import torch
import gc

class foo:

    def get_model(model_path):

        model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )

        return model
    
    def run_voice_design(model, persona):
     
        # 2. Run the specific design method
        with torch.no_grad():
            # Using your specific method here
            ref_wavs, sr = model.generate_voice_design(
                text=persona.ref_text,
                language=persona.language,
                instruct=persona.instruct
            )
        
        return ref_wavs, sr

    def run_voice_clone(model, ref_audio_tuple, target_text):
        
        # 2. Run the specific clone method
         with torch.no_grad():
            # Using your second specific method here
            clone_prompt = model.create_voice_clone_prompt(
                ref_audio=ref_audio_tuple,
                ref_text=target_text
            )

""" # --- Execution Flow ---

# Step 1: Design the voice
design_audio, sampling_rate = run_voice_design("/workspace/qwen/models/...VoiceDesign", "Indian Hindu Woman")

# Step 2: Clear GPU completely
gc.collect()
torch.cuda.empty_cache()

# Step 3: Use the Clone model
final_prompt = run_voice_clone("/workspace/qwen/models/...Base", "/workspace/audio/input/...", "Target text") """
