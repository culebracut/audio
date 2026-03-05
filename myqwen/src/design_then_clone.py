import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import gc
from persona import Persona 
from run_inference import foo 

sentences = [
    "Four score and seven years ago our fathers brought forth on this continent a new nation, conceived in liberty, and dedicated to the proposition that all men are created equal.",
    "Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battlefield of that war.",
    "We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live.",
    "It is altogether fitting and proper that we should do this.",
    "But, in a larger sense, we can not dedicate -- we can not consecrate -- we can not hallow -- this ground.",
    "The brave  men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract.",
    "The world will little note, nor long remember what we say here, but it can never forget what they did here.",
    "It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced.",
    "It is rather for us to be here dedicated to the great task remaining before us -- that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion -- that we here highly resolve that these dead shall not have died in vain -- that this nation, under God, shall have a new birth of freedom -- and that government of the people, by the people, for the people, shall not perish from the earth.",
]

# ref_text = "H-hey! You dropped your... uh... calculus notebook? I mean, I think it's yours? Maybe?"
# ref_instruct = "Male, 17 years old, tenor range, gaining confidence - deeper breath support now, though vowels still tighten when nervous"
#     "instruct":[
#     "A deep, gravelly, rhythmic Western drawl, with a weathered, gravelly tone, like a seasoned cowboy who's spent years under the sun, with a slow, deliberate delivery that emphasizes the rugged, authoritative grit of a classic Western hero, punctuated by occasional pauses that add to the dramatic effect, and a weary, authoritative grit that conveys the hard-earned wisdom and resilience, like John Wayne himself."
#     ],

# sentences = [
#     "No problem! I actually... kinda finished those already? If you want to compare answers or something...",
#     "What? No! I mean yes but not like... I just think you're... your titration technique is really precise!",
# ]

output_dir = "/workspace/audio/output/"
# create a reference audio in the target style using the VoiceDesign model

# 1. Your raw data
persona_data = {
    "id":"duke",
    "description":"John Wayne",
    "seed": 100,
    "temp": 0.9,
    "language": "English",
    "instruct":[
    "Male, 17 years old, tenor range, gaining confidence - deeper breath support now, though vowels still tighten when nervous"
    ],
    "ref_audio": "/workspace/audio/input/JohnWayne/JW.wav",
    "ref_text": "This that the White Man calls charity is a fine thing for widows and orphans, but no warrior can accept it, for if he does, he is no longer a man and when he is no longer a man, he is nothing and better off dead."
    }

# 2. Creating an object
persona = Persona.from_dict(persona_data)

# get a model instance
model_path = "/workspace/qwen/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
design_model = foo.get_model(model_path)

# design the voice and get the reference audio
wavs,sr = foo.run_voice_design(design_model, persona)

###################
# 1. Manually break the reference
print(f"GPU Memory before clearing: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
if 'design_model' in locals():
    # If .cpu() is missing, we skip it and go straight to deletion
    del design_model

# 2. Crucial: Clear any 'ghost' references in the Python Garbage Collector
#for _ in range(3): # Running it a few times ensures nested tensors are caught
gc.collect()

# 3. Force the GPU to release the memory back to the OS
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# 4. Verification (Optional but helpful)
print(f"GPU Memory after clearing: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
###################


                # FREE UP GPU MEMORY
#model_path = "/workspace/qwen/models/Qwen3-TTS-12Hz-1.7B-Base"
model_path = "/workspace/qwen/models/Qwen3-TTS-12Hz-0.6B-Base"
clone_model = foo.get_model(model_path)

# build a reusable clone prompt from the voice design reference
#voice_clone_prompt = foo.run_voice_clone(clone_model, persona.ref_audio, persona.ref_text)

voice_clone_prompt = clone_model.create_voice_clone_prompt(
    ref_audio=(wavs[0], sr), 
    ref_text=persona.ref_text
)

# reuse it for multiple single calls
wavs, sr = clone_model.generate_voice_clone(
    text=persona.ref_text,
    language="English",
    voice_clone_prompt=voice_clone_prompt,
)
sf.write(output_dir + "clone_single_0.wav", wavs[0], sr)

# reuse it for multiple single calls
wavs, sr = clone_model.generate_voice_clone(
    text=sentences[0],  
    language="English",
    voice_clone_prompt=voice_clone_prompt,
)
sf.write(f"{output_dir}clone_single_1.wav", wavs[0], sr)

wavs, sr = clone_model.generate_voice_clone(
    text=sentences[1],
    language="English",
    voice_clone_prompt=voice_clone_prompt,
)
sf.write(output_dir + "clone_single_2.wav", wavs[0], sr)

# or batch generate in one call
wavs, sr = clone_model.generate_voice_clone(
    text=sentences,
    language=["English"]*len(sentences),
    voice_clone_prompt=voice_clone_prompt,
)
for i, w in enumerate(wavs):
    sf.write(output_dir + f"clone_batch_{i}.wav", w, sr)


###################
# 1. Manually break the reference
print(f"GPU Memory before clearing: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
if 'design_model' in locals():
    # If .cpu() is missing, we skip it and go straight to deletion
    del design_model

# 2. Crucial: Clear any 'ghost' references in the Python Garbage Collector
for _ in range(3): # Running it a few times ensures nested tensors are caught
    gc.collect()

# 3. Force the GPU to release the memory back to the OS
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# 4. Verification (Optional but helpful)
print(f"GPU Memory after clearing: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
###################




