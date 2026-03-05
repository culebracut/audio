"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              Qwen3 Voice Assistant - Full Speech Pipeline                    ║
║                                                                              ║
║  Flow:  🎤 Microphone  →  Qwen3-ASR-0.6B  →  Qwen3-0.6B LLM               ║
║                        →  Qwen3-TTS-0.6B  →  🔊 Speaker                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Models used (all downloadable, all 0.6B where available):
  • Speech-to-Text : Qwen/Qwen3-ASR-0.6B       (~600 MB)
  • Language Model : Qwen/Qwen3-0.6B            (~1.2 GB)
  • Text-to-Speech : Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice (~600 MB)

Requirements  →  see requirements.txt
Hardware      →  NVIDIA GPU with ≥6 GB VRAM recommended (CPU also works, slower)
"""

import os
import sys
import time
import logging
import argparse
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("voice_assistant")


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
class Config:
    # ── ASR (Speech-to-Text) ──────────────────────────────────────────────
    ASR_MODEL        = "Qwen/Qwen3-ASR-0.6B"
    ASR_LANGUAGE     = None          # None = auto-detect; or e.g. "English"
    ASR_MAX_TOKENS   = 256

    # ── LLM (Question Answering) ──────────────────────────────────────────
    LLM_MODEL        = "Qwen/Qwen3-0.6B"
    LLM_MAX_TOKENS   = 512
    LLM_TEMPERATURE  = 0.7
    LLM_SYSTEM_MSG   = (
        "You are a helpful, concise voice assistant. "
        "Keep answers short (2-4 sentences) and conversational, "
        "since your response will be spoken aloud."
    )

    # ── TTS (Text-to-Speech) ──────────────────────────────────────────────
    TTS_MODEL        = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    TTS_SPEAKER      = "Ryan"        # Sunny American male voice
    TTS_LANGUAGE     = "English"
    TTS_INSTRUCT     = ""            # e.g. "Speak in a warm, friendly tone"

    # ── Recording ────────────────────────────────────────────────────────
    RECORD_SAMPLERATE = 16_000       # Hz – ideal for ASR
    RECORD_CHANNELS   = 1
    RECORD_DTYPE      = "int16"
    SILENCE_THRESHOLD = 500          # RMS below this = silence
    SILENCE_SECONDS   = 2.0          # seconds of silence to stop recording
    MAX_RECORD_SECS   = 30           # safety cap

    # ── Output ───────────────────────────────────────────────────────────
    OUTPUT_DIR       = Path("./output_audio")
    DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE            = torch.bfloat16 if torch.cuda.is_available() else torch.float32


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def banner(text: str, char: str = "─") -> None:
    width = 72
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def rms(data: np.ndarray) -> float:
    """Root-mean-square amplitude of audio chunk."""
    return float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 1 – RECORD AUDIO FROM MICROPHONE
# ═════════════════════════════════════════════════════════════════════════════
def record_question(cfg: Config) -> np.ndarray:
    """
    Record from the default microphone until silence is detected or the
    maximum recording time is reached.

    Returns
    -------
    np.ndarray  int16 mono audio at cfg.RECORD_SAMPLERATE
    """
    banner("🎤  Listening  – speak your question, then stay silent for "
           f"{cfg.SILENCE_SECONDS:.0f}s to finish", "═")

    chunk_size  = int(cfg.RECORD_SAMPLERATE * 0.1)   # 100 ms chunks
    max_chunks  = int(cfg.MAX_RECORD_SECS / 0.1)
    silence_cap = int(cfg.SILENCE_SECONDS / 0.1)

    frames: list[np.ndarray] = []
    silence_count = 0
    recording_started = False

    print("  [Waiting for speech...]", end="", flush=True)

    with sd.InputStream(
        samplerate=cfg.RECORD_SAMPLERATE,
        channels=cfg.RECORD_CHANNELS,
        dtype=cfg.RECORD_DTYPE,
        blocksize=chunk_size,
    ) as stream:
        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_size)
            chunk_flat = chunk.flatten()
            level = rms(chunk_flat)

            if level > cfg.SILENCE_THRESHOLD:
                if not recording_started:
                    print("\r  [Recording...      ]", end="", flush=True)
                    recording_started = True
                frames.append(chunk_flat)
                silence_count = 0
            elif recording_started:
                frames.append(chunk_flat)  # keep trailing silence for context
                silence_count += 1
                dots = "." * min(silence_count, silence_cap)
                print(f"\r  [Silence{dots:<{silence_cap}}]", end="", flush=True)
                if silence_count >= silence_cap:
                    break

    print()  # newline after inline status
    if not frames:
        raise RuntimeError("No speech detected – please check your microphone.")

    audio = np.concatenate(frames)
    duration = len(audio) / cfg.RECORD_SAMPLERATE
    log.info("Recorded %.1f seconds of audio (%d samples)", duration, len(audio))
    return audio


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 2 – TRANSCRIBE WITH QWEN3-ASR
# ═════════════════════════════════════════════════════════════════════════════
def load_asr_model(cfg: Config):
    """Load (and cache) the Qwen3-ASR model."""
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError:
        log.error(
            "qwen-asr not installed. Run:  pip install qwen-asr\n"
            "  (or: git clone https://github.com/QwenLM/Qwen3-ASR && cd Qwen3-ASR && pip install -e .)"
        )
        sys.exit(1)

    log.info("Loading ASR model: %s  [device=%s]", cfg.ASR_MODEL, cfg.DEVICE)
    device_map = cfg.DEVICE if cfg.DEVICE == "cpu" else f"cuda:0"
    model = Qwen3ASRModel.from_pretrained(
        cfg.ASR_MODEL,
        dtype=cfg.DTYPE,
        device_map=device_map,
        max_new_tokens=cfg.ASR_MAX_TOKENS,
    )
    log.info("ASR model loaded ✓")
    return model


def transcribe(asr_model, audio: np.ndarray, cfg: Config) -> str:
    """
    Transcribe a numpy audio array to text.

    The qwen_asr package accepts a (np.ndarray, sample_rate) tuple directly,
    so we avoid writing a temporary file.
    """
    banner("📝  Transcribing speech → text  (Qwen3-ASR-0.6B)")

    audio_input = (audio, cfg.RECORD_SAMPLERATE)
    results = asr_model.transcribe(audio=audio_input, language=cfg.ASR_LANGUAGE)
    result  = results[0]

    lang = getattr(result, "language", "unknown")
    text = result.text.strip()
    log.info("Detected language: %s", lang)
    print(f"\n  Transcription: \"{text}\"")
    return text


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 3 – ANSWER WITH QWEN3-0.6B LLM
# ═════════════════════════════════════════════════════════════════════════════
def load_llm(cfg: Config):
    """Load (and cache) the Qwen3 language model + tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading LLM: %s  [device=%s]", cfg.LLM_MODEL, cfg.DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(cfg.LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.LLM_MODEL,
        torch_dtype=cfg.DTYPE,
        device_map="auto",
    )
    model.eval()
    log.info("LLM loaded ✓")
    return model, tokenizer


def answer_question(llm_model, tokenizer, question: str, cfg: Config) -> str:
    """Generate an answer for *question* using the Qwen3-0.6B model."""
    banner("🤖  Generating answer  (Qwen3-0.6B)")

    messages = [
        {"role": "system", "content": cfg.LLM_SYSTEM_MSG},
        {"role": "user",   "content": question},
    ]

    # Use enable_thinking=False for concise, fast non-reasoning replies.
    # Set to True for deeper reasoning on complex questions.
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    device = next(llm_model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        output_ids = llm_model.generate(
            **inputs,
            max_new_tokens=cfg.LLM_MAX_TOKENS,
            temperature=cfg.LLM_TEMPERATURE,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    # Strip the prompt tokens from the output
    new_ids  = output_ids[0][len(inputs.input_ids[0]):]
    answer   = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    tokens_generated = len(new_ids)
    log.info(
        "Answer generated in %.1fs  (%d tokens, %.1f tok/s)",
        elapsed, tokens_generated, tokens_generated / elapsed,
    )
    print(f"\n  Answer: \"{answer}\"")
    return answer


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 4 – SYNTHESISE SPEECH WITH QWEN3-TTS
# ═════════════════════════════════════════════════════════════════════════════
def load_tts_model(cfg: Config):
    """Load (and cache) the Qwen3-TTS model."""
    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError:
        log.error(
            "qwen-tts not installed. Run:  pip install qwen-tts\n"
            "  (or: git clone https://github.com/QwenLM/Qwen3-TTS && cd Qwen3-TTS && pip install -e .)"
        )
        sys.exit(1)

    log.info("Loading TTS model: %s  [device=%s]", cfg.TTS_MODEL, cfg.DEVICE)
    device_map = "cpu" if cfg.DEVICE == "cpu" else "cuda:0"

    # flash_attention_2 only on CUDA; fall back to default on CPU
    attn = "flash_attention_2" if cfg.DEVICE == "cuda" else "eager"

    model = Qwen3TTSModel.from_pretrained(
        cfg.TTS_MODEL,
        device_map=device_map,
        dtype=cfg.DTYPE,
        attn_implementation=attn,
    )
    log.info("TTS model loaded ✓")
    return model


def synthesise_speech(tts_model, text: str, cfg: Config) -> tuple[np.ndarray, int]:
    """
    Convert *text* to audio waveform.

    Returns
    -------
    (waveform: np.ndarray, sample_rate: int)
    """
    banner("🔊  Synthesising speech  (Qwen3-TTS-0.6B-CustomVoice)")

    instruct = cfg.TTS_INSTRUCT or None
    wavs, sr = tts_model.generate_custom_voice(
        text=text,
        language=cfg.TTS_LANGUAGE,
        speaker=cfg.TTS_SPEAKER,
        instruct=instruct,
    )
    log.info("TTS synthesis complete – sample rate: %d Hz, samples: %d", sr, len(wavs[0]))
    return wavs[0], sr


# ═════════════════════════════════════════════════════════════════════════════
#  PLAYBACK + SAVE
# ═════════════════════════════════════════════════════════════════════════════
def play_audio(waveform: np.ndarray, sample_rate: int) -> None:
    """Play audio through the default output device."""
    banner("▶  Playing response audio")
    # sounddevice expects float32 in [-1, 1]
    audio_f32 = waveform.astype(np.float32)
    if audio_f32.max() > 1.0:
        audio_f32 /= np.abs(audio_f32).max()
    sd.play(audio_f32, samplerate=sample_rate)
    sd.wait()
    print("  Playback complete.")


def save_audio(waveform: np.ndarray, sample_rate: int, cfg: Config) -> Path:
    """Save the synthesised audio to a timestamped WAV file."""
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = cfg.OUTPUT_DIR / f"response_{ts}.wav"
    sf.write(str(path), waveform, sample_rate)
    log.info("Audio saved → %s", path)
    return path


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
class VoiceAssistant:
    """
    Orchestrates the full pipeline:
        microphone  →  ASR  →  LLM  →  TTS  →  speaker
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.asr_model   = None
        self.llm_model   = None
        self.tokenizer   = None
        self.tts_model   = None

    # ── lazy-load models on first use ─────────────────────────────────────
    def _ensure_asr(self):
        if self.asr_model is None:
            self.asr_model = load_asr_model(self.cfg)

    def _ensure_llm(self):
        if self.llm_model is None:
            self.llm_model, self.tokenizer = load_llm(self.cfg)

    def _ensure_tts(self):
        if self.tts_model is None:
            self.tts_model = load_tts_model(self.cfg)

    # ── single Q&A turn ───────────────────────────────────────────────────
    def run_once(self, audio_path: str | None = None) -> dict:
        """
        Run one complete question → answer cycle.

        Parameters
        ----------
        audio_path : str or None
            If supplied, use this WAV file as input instead of recording.

        Returns
        -------
        dict  with keys: question, answer, audio_path
        """
        cfg = self.cfg
        t_start = time.time()

        # ── 1. Input audio ────────────────────────────────────────────────
        if audio_path:
            log.info("Reading audio from file: %s", audio_path)
            audio, sr = sf.read(audio_path, dtype="int16", always_2d=False)
            if sr != cfg.RECORD_SAMPLERATE:
                import resampy
                audio = resampy.resample(
                    audio.astype(np.float32), sr, cfg.RECORD_SAMPLERATE
                ).astype(np.int16)
        else:
            audio = record_question(cfg)

        # ── 2. Transcribe ─────────────────────────────────────────────────
        self._ensure_asr()
        question = transcribe(self.asr_model, audio, cfg)

        if not question:
            log.warning("Empty transcription – skipping LLM + TTS.")
            return {"question": "", "answer": "", "audio_path": None}

        # ── 3. Answer ─────────────────────────────────────────────────────
        self._ensure_llm()
        answer = answer_question(self.llm_model, self.tokenizer, question, cfg)

        # ── 4. Synthesise speech ──────────────────────────────────────────
        self._ensure_tts()
        waveform, sample_rate = synthesise_speech(self.tts_model, answer, cfg)

        # ── 5. Save & play ────────────────────────────────────────────────
        out_path = save_audio(waveform, sample_rate, cfg)
        play_audio(waveform, sample_rate)

        elapsed = time.time() - t_start
        banner(f"✅  Pipeline complete in {elapsed:.1f}s  |  saved: {out_path}", "═")
        return {"question": question, "answer": answer, "audio_path": str(out_path)}

    # ── interactive loop ──────────────────────────────────────────────────
    def run_interactive(self):
        """Keep asking until the user says 'quit' or presses Ctrl-C."""
        banner("Qwen3 Voice Assistant – Interactive Mode", "═")
        print(
            "  Say a question → get a spoken answer.\n"
            "  Say 'quit', 'exit', or 'bye' to stop.\n"
            "  Press Ctrl-C at any time to exit.\n"
        )
        while True:
            try:
                result = self.run_once()
                q = result.get("question", "").lower()
                if any(kw in q for kw in ("quit", "exit", "bye", "goodbye")):
                    print("\n  Goodbye! 👋")
                    break
            except KeyboardInterrupt:
                print("\n\n  Interrupted by user. Exiting.")
                break
            except Exception as exc:
                log.error("Error during pipeline: %s", exc, exc_info=True)
                print("  An error occurred. Try again or press Ctrl-C to quit.")


# ═════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen3 Voice Assistant – full speech pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", "-i", default=None, metavar="FILE",
        help="Path to an existing WAV/audio file to use instead of recording.",
    )
    p.add_argument(
        "--interactive", "-I", action="store_true",
        help="Run in interactive loop mode (keep asking questions).",
    )
    p.add_argument(
        "--asr-model", default=Config.ASR_MODEL,
        help="Hugging Face model ID for ASR.",
    )
    p.add_argument(
        "--llm-model", default=Config.LLM_MODEL,
        help="Hugging Face model ID for the language model.",
    )
    p.add_argument(
        "--tts-model", default=Config.TTS_MODEL,
        help="Hugging Face model ID for TTS.",
    )
    p.add_argument(
        "--speaker", default=Config.TTS_SPEAKER,
        choices=["Vivian", "Ryan", "Ethan", "Chelsie", "Chloe", "Dylan",
                 "Aria", "Yui", "Jisoo"],
        help="TTS preset speaker voice.",
    )
    p.add_argument(
        "--language", default=Config.TTS_LANGUAGE,
        help="Language for TTS synthesis (e.g. English, Chinese, Japanese).",
    )
    p.add_argument(
        "--tts-instruct", default=Config.TTS_INSTRUCT, metavar="INSTRUCT",
        help="Natural-language style instruction for the TTS voice.",
    )
    p.add_argument(
        "--max-record", type=float, default=Config.MAX_RECORD_SECS,
        help="Maximum seconds to record from the microphone.",
    )
    p.add_argument(
        "--output-dir", default=str(Config.OUTPUT_DIR), metavar="DIR",
        help="Directory to save output WAV files.",
    )
    p.add_argument(
        "--no-play", action="store_true",
        help="Skip audio playback (only save to file).",
    )
    p.add_argument(
        "--device", default=None, choices=["cpu", "cuda"],
        help="Force a specific compute device.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    cfg = Config()
    cfg.ASR_MODEL      = args.asr_model
    cfg.LLM_MODEL      = args.llm_model
    cfg.TTS_MODEL      = args.tts_model
    cfg.TTS_SPEAKER    = args.speaker
    cfg.TTS_LANGUAGE   = args.language
    cfg.TTS_INSTRUCT   = args.tts_instruct
    cfg.MAX_RECORD_SECS = args.max_record
    cfg.OUTPUT_DIR     = Path(args.output_dir)
    if args.device:
        cfg.DEVICE = args.device
        cfg.DTYPE  = torch.bfloat16 if args.device == "cuda" else torch.float32

    assistant = VoiceAssistant(cfg)

    if args.interactive:
        # Monkeypatch play_audio to a no-op if --no-play
        if args.no_play:
            import voice_assistant as _self_mod
            _self_mod.play_audio = lambda *a, **k: print("  [Playback skipped]")
        assistant.run_interactive()
    else:
        if args.no_play:
            import voice_assistant as _self_mod
            _self_mod.play_audio = lambda *a, **k: print("  [Playback skipped]")
        assistant.run_once(audio_path=args.input)


if __name__ == "__main__":
    main()
