import torch
import numpy as np
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# 1. Load the Model
# Ensure bfloat16 for Flash-Attention 2 and specify the implementation
model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = Qwen3TTSModel.from_pretrained(
    model_id, 
    device_map=device,
    dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2")

# 2. Device Placement
# Since the wrapper lacks .to(), we move the internal PyTorch module
# model.model.to("cuda")
# print("Internal model moved to 3060 Ti.")

# 3. Generation (Base Model / Voice Cloning)
# For the Base model, you must provide a reference audio or use the voice clone method
myWav = "data/audio/input/MobyDick/moby_consider.wav"
myWavInputText = "Consider all this; and then turn to this green, gentle, and most docile earth; consider them both, the sea and the land; and do you not find a strange analogy to something in yourself? For as this appalling ocean surrounds the verdant land, so in the soul of man there lies one insular Tahiti, full of peace and joy, but encompassed by all the horrors of the half known life. God keep thee! Push not off from that isle, thou canst never return! "
myPrompt = model.create_voice_clone_prompt(
    ref_audio=myWav, # Path to your 5-10s clip
    ref_text=myWavInputText,
    x_vector_only_mode=False
)

print("--- Generating audio ---")
#text_prompt = "Slap some bacon on a biscuit and let's go! We're burnin' daylight!"
text_prompt = "Oh my god, you literally will not believe what just happened because I am actually shaking and screaming right now! So basically, I was just vibing and minding my own business when I saw the news and my jaw honestly dropped to the floor because it is so insane. I am so hyped that I think I might actually explode and I seriously cannot breathe because this is the best thing that has ever happened in my entire life. Like, is this even real life right now or am I dreaming because there is no way this is actually happening for real. I am literally obsessed and we seriously have to go celebrate right this second because I am losing my mind and I just cannot even deal with how amazing this is! Let's gooooo!"
try:
    # Use the specific method for the Base/Voice-Clone model
    wavs, sr = model.generate_voice_clone(
        text=text_prompt,
        language="English",
        device=device,
        voice_clone_prompt=myPrompt
    )

    #wav = np.asarray(wavs[0], dtype=np.float32)
    # 2. Fix the shape error: Move to CPU and detach from the graph
    # If wavs[0] is a tensor, we need .cpu().numpy()
    if torch.is_tensor(wavs[0]):
        wav = wavs[0].cpu().numpy().flatten()
    else:
        wav = np.array(wavs[0]).flatten()

    # 4. Save to file
    # audio_data is a numpy array returned by the helper
    outputWavPath = "data/audio/output/output.wav"
    sf.write(outputWavPath, wav, samplerate=24000)
    print("--- Success! Saved to output.wav ---")

except Exception as e:
    print(f"Generation failed: {e}")
