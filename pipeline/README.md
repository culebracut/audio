# Qwen3 Voice Assistant 🎤 → 🤖 → 🔊

A fully local, GPU-friendly **speech-in / speech-out** pipeline built entirely on Qwen3 models.

```
Microphone  ──►  Qwen3-ASR-0.6B  ──►  Qwen3-0.6B LLM  ──►  Qwen3-TTS-0.6B  ──►  Speaker
  (record)       (speech-to-text)      (Q&A / reasoning)    (text-to-speech)    (playback)
```

---

## Models...

| Stage | Model | Size | Downloaded from |
|-------|-------|------|-----------------|
| Speech recognition | `Qwen/Qwen3-ASR-0.6B` | ~600 MB | HuggingFace / ModelScope |
| Language model | `Qwen/Qwen3-0.6B` | ~1.2 GB | HuggingFace / ModelScope |
| Speech synthesis | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | ~600 MB | HuggingFace / ModelScope |

All models are downloaded **automatically** on first run.

---

## Requirements

- Python **3.10+** (3.12 recommended)
- NVIDIA GPU with **≥6 GB VRAM** (recommended) — or CPU (slower)
- Working microphone & audio output

---

## Installation

### 1 · Create a clean environment

```bash
conda create -n qwen3-va python=3.12 -y
conda activate qwen3-va
```

### 2 · Install Python dependencies

```bash
pip install -r requirements.txt
```

> **GPU speed-up (optional)** – FlashAttention 2 cuts VRAM usage and speeds up generation:
> ```bash
> pip install flash-attn --no-build-isolation
> ```
> Requires NVIDIA Ada Lovelace / Ampere / Hopper (RTX 30xx / 40xx / A100 / H100).

### 3 · System audio libraries

#### Ubuntu / Debian
```bash
sudo apt update && sudo apt install -y libportaudio2 libsndfile1
```

#### macOS
```bash
brew install portaudio libsndfile
```

#### Windows
PortAudio is bundled with the `sounddevice` wheel — no extra step needed.

---

## Usage

### Single question (microphone)

```bash
python voice_assistant.py
```

Speak your question → stay silent for 2 seconds → hear the spoken answer.

### Interactive loop

```bash
python voice_assistant.py --interactive
```

Keeps asking for questions until you say *"quit"*, *"exit"*, or *"bye"*.

### Use a pre-recorded audio file

```bash
python voice_assistant.py --input my_question.wav
```

### All CLI options

```
python voice_assistant.py --help

  --input FILE          Use this WAV file instead of recording (default: record)
  --interactive         Keep looping until you say quit/exit/bye
  --asr-model MODEL     ASR model ID  (default: Qwen/Qwen3-ASR-0.6B)
  --llm-model MODEL     LLM model ID  (default: Qwen/Qwen3-0.6B)
  --tts-model MODEL     TTS model ID  (default: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
  --speaker NAME        TTS preset voice  (default: Ryan)
  --language LANG       TTS output language  (default: English)
  --tts-instruct TEXT   Natural-language voice style instruction
  --max-record SECS     Max recording seconds  (default: 30)
  --output-dir DIR      Where to save output WAV files  (default: ./output_audio)
  --no-play             Save audio only, skip playback
  --device {cpu,cuda}   Force device  (default: auto-detect)
```

---

## Available TTS Speakers

| Name | Description |
|------|-------------|
| **Ryan** | Sunny American male, clear midrange |
| **Vivian** | Bright, slightly edgy young female |
| **Chelsie** | Warm, gentle young female |
| **Ethan** | Seasoned male, low and mellow |
| **Chloe** | Dynamic female with rhythmic drive |
| **Dylan** | Youthful Beijing male, natural timbre |
| **Aria** | Warm Korean female with rich emotion |
| **Yui** | Playful Japanese female, light timbre |

---

## Upgrade to 1.7B models

If you have a more powerful GPU (≥12 GB VRAM), swap the 0.6B models for their larger siblings:

```bash
python voice_assistant.py \
  --asr-model Qwen/Qwen3-ASR-1.7B \
  --llm-model Qwen/Qwen3-1.7B \
  --tts-model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

> **Note on TTS quality:** The 0.6B TTS CustomVoice produces good quality for short to medium text.
> For voice cloning instead of preset voices, use the `*-Base` TTS model variants.

---

## Project structure

```
qwen3_voice_assistant/
├── voice_assistant.py   ← main pipeline (this file)
├── requirements.txt     ← Python dependencies
└── README.md            ← this file
output_audio/            ← saved WAV responses (auto-created)
```

---

## Licence

All Qwen3 model weights are released under the **Apache 2.0** licence by Alibaba Cloud / Qwen Team.
