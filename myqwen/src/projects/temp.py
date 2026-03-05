from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

try:
    # trust_remote_code=True is mandatory for new architectures like Qwen3-TTS
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    print("✅ Model loaded successfully using custom remote code!")
except Exception as e:
    print(f"❌ Still failing: {e}")
