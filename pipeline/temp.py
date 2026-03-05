from pyexpat import model

import torch
from qwen_asr import Qwen3ASRModel
from qwen_tts import Qwen3TTSModel
import sounddevice as sd

#model_path = "Qwen/Qwen3-ASR-1.7B"
model_path = "Qwen/Qwen3-ASR-0.6B"
asr_model = Qwen3ASRModel.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="flash_attention_2",
    max_inference_batch_size=4, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
    max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
)

ref_audio = "src/moby_consider.wav"
transcribed_text = asr_model.transcribe(
    audio= ref_audio, # can also be a list of audio files
    language=None, # set "English" to force the language
)

ref_language = transcribed_text[0].language
ref_text = transcribed_text[0].text

print(ref_language)
print(ref_text)

#model_path = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
model_path = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
tts_model = Qwen3TTSModel.from_pretrained(
    model_path,
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    #max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
    #max_new_tokens=1024, # Maximum number of tokens to generate. Set a larger value for long audio output.
)

""" with torch.no_grad():
    generated_audio = tts_model.voice_clone(
        reference_audio=filepath, # Parrots the original speaker's voice
        text=transcribed_text
    ) """

""" wavs, sr = tts_model.generate_voice_clone(
    text="I am solving the equation: x = [-b ± √(b²-4ac)] / 2a? Nobody can — it's a disaster (◍•͈⌔•͈◍), very sad!",
    language=ref_language,
    ref_audio=ref_audio,
    ref_text=ref_text,
) """

my_audio="/home/system/workspace/qwen/myqwen/data/audio/input/JohnWayne/JW.wav"
my_text="This that the White Man calls charity is a fine thing for widows and orphans, but no warrior can accept it, for if he does, he is no longer a man and when he is no longer a man, he is nothing and better off dead."
wavs, sr = tts_model.generate_voice_clone(
    text=ref_text,
    language=ref_language,
    ref_audio=my_audio,
    ref_text=my_text,
)

# sd.play(generated_audio.cpu().numpy(), samplerate=24000)
sd.play(wavs[0], samplerate=sr)
sd.wait()