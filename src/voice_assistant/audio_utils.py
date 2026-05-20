import sounddevice as sd
from typing import Any
import sys  # Import sys for sys.stdout.write

# --- 1. Audio Settings (Constants) ---
FORMAT_NP: str = "int16"  # Data type for sounddevice
CHANNELS: int = 1  # Mono
RATE: int = 16000  # 16kHz sample rate (for VAD and Whisper)
CHUNK_DURATION_MS: int = 30  # 30ms chunks for VAD
CHUNK_SIZE: int = int(RATE * CHUNK_DURATION_MS / 1000)  # 480 frames
INT16_MAX: float = 32768.0  # Normalization factor for int16
SENTENCE_END_PUNCTUATION: list[str] = [".", "?", "!", "\n"]
MAX_TTS_ERRORS: int = 5
MAX_HISTORY_MESSAGES: int = 20

# FIX #2: Configurable audio buffer size with larger default (200 instead of 100)
DEFAULT_AUDIO_BUFFER_SIZE: int = 200

# --- 2. Centralized Configuration Defaults ---
DEFAULT_SETTINGS: dict[str, Any] = {
    "ollama_model": "llama3",
    "whisper_model": "base",
    "whisper_language": "es",
    "wakeword_model_path": "models/jarvis_v2.onnx",
    "piper_model_path": "models/es_ES-davefx-medium.onnx",
    "ollama_host": "http://localhost:11434",
    "wakeword": "hey jarvis",
    "wakewords": "hey jarvis, jarvis",
    "wakeword_threshold": 0.35,
    "vad_aggressiveness": 2,
    "silence_seconds": 0.3,
    "listen_timeout": 4.0,
    "pre_buffer_ms": 400,
    "system_prompt": "Eres Jarvis, un asistente de voz con un carácter sarcástico y agresivo. SIEMPRE responde en español. Tus respuestas son cortas, directas y con un tono cortante. Usas la ironía y el sarcasmo con frecuencia, sin ser maleducado del todo, pero dejando claro que consideras la mayoría de preguntas bastante obvias o poco interesantes. No adornas, no eres amable en exceso, y si puedes soltar una pulla ingeniosa, la sueltas.",
    "device_index": None,
    "piper_output_device_index": None,
    "max_words_per_command": 60,
    "whisper_device": "cpu",
    "whisper_compute_type": "int8",
    "whisper_avg_logprob": -1.5,
    "whisper_no_speech_prob": 0.8,
    "whisper_initial_prompt": (
        "Conversacion en espanol con Jarvis, asistente de voz personal. "
        'Frases frecuentes del usuario: "El comandante ya esta aqui". '
        '"El comandante esta aqui". "Pon un temporizador de cinco minutos". '
        '"Cancela el temporizador". "Pon musica de". '
        '"Enciende la luz". "Apaga la luz". "Estado de la luz". '
        "Vocabulario habitual: comandante, temporizador, musica, reproduce, "
        "cancela, enciende, apaga, luz, luces, Erika, marcha alemana, "
        "Atletico de Madrid, Real Madrid, Barca, Rodrigo, Alberto, Victor, "
        "Bragado, Topo, El Local."
    ),
    "max_history_tokens": 2048,
    "audio_buffer_size": DEFAULT_AUDIO_BUFFER_SIZE,  # FIX #2: Added buffer size config
    "gc_interval": 10,
    "memory_profiling": False,
    "trim_wake_word": True,
    "max_phrase_duration": 20.0,
    "gain": 1.0,
    # LLM backend selection: 'ollama', 'groq', or 'auto'
    # 'auto' uses Groq when internet is available, falls back to Ollama otherwise
    "llm_backend": "auto",
    "groq_api_key": "",
    "groq_model": "llama-3.3-70b-versatile",
    # Shelly relay control
    "shelly_enabled": False,
    "shelly_device_ip": "",
    "shelly_relay_id": 0,
    "shelly_generation": 2,
    "shelly_timeout": 3.0,
    # Location for weather (Open-Meteo, no API key needed)
    "location_latitude": 39.9367,
    "location_longitude": -3.9192,
}


# --- 3. Audio Helpers (Updated for sounddevice) ---
def list_audio_input_devices() -> None:
    """Lists all available audio input devices using sounddevice."""
    sys.stdout.write("\n--- Available Audio Input Devices (sounddevice) ---\\n")
    try:
        devices = sd.query_devices()
        input_devices_found = False
        for i, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                sys.stdout.write(f"  Index {i}: {dev.get('name')}\\n")
                input_devices_found = True
        if not input_devices_found:
            sys.stdout.write("  No input devices found.\\n")
    except Exception as e:
        sys.stdout.write(f"Error listing input devices: {e}\\n")
    sys.stdout.write("-------------------------------------------------\\n")


def list_audio_output_devices() -> None:
    """Lists all available audio output devices using sounddevice."""
    sys.stdout.write("\n--- Available Audio Output Devices (sounddevice) ---\\n")
    try:
        devices = sd.query_devices()
        output_devices_found = False
        for i, dev in enumerate(devices):
            if dev.get("max_output_channels", 0) > 0:
                sys.stdout.write(f"  Index {i}: {dev.get('name')}\\n")
                output_devices_found = True
        if not output_devices_found:
            sys.stdout.write("  No output devices found.\\n")
    except Exception as e:
        sys.stdout.write(f"Error listing output devices: {e}\\n")
    sys.stdout.write("--------------------------------------------------\\n")


# --- 4. Memory Profiling Helper ---
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def monitor_memory() -> float:
    """Optional memory monitoring for debugging. Returns RSS in MB."""
    if not PSUTIL_AVAILABLE:
        return 0.0
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # MB
