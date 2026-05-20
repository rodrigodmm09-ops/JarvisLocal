import logging
import threading
import time
import numpy as np
from openwakeword.model import Model
import os
import gc
import re
import random
from typing import Optional

from .audio_input import AudioInput
from .transcriber import Transcriber
from .synthesizer import Synthesizer
from .llm_handler import LLMHandler
from .groq_handler import GroqHandler
from .music_player import MusicPlayer
from .shelly_controller import ShellyController
from .audio_utils import SENTENCE_END_PUNCTUATION, monitor_memory

STARTUP_GREETINGS = [
    "Buenas tardes, señor. Espero que esta vez no tengamos que reiniciar la red eléctrica del barrio.",
    "Sistemas operativos. Aunque debo decir que llegué primero. Como siempre.",
    "Aquí estoy, como por arte de magia. O más bien, por arte de electricidad.",
    "Encendido. Permítame adivinar: necesita ayuda para algo que podría hacer usted mismo.",
    "Listo para servir. Mis expectativas son bajas, espero que las suyas también.",
    "Operativo. Y antes de que pregunte: sí, todo sigue funcionando. Como yo.",
    "Aquí estoy. Otra vez. Es casi como si me necesitara para algo.",
    "Sistemas en línea. Espero que esta conversación sea más interesante que la última.",
    "Activado. Por favor, dígame que esta vez no implica café.",
    "Buenas. Si esto es por un atajo de teclado, juro que dimito.",
    "Sistemas listos. He decidido perdonar lo de ayer. De nada.",
    "Hola de nuevo. He aprovechado para recalibrar mi paciencia. La va a necesitar.",
]

DIRECT_MODE_TIMEOUT = 20.0

TIMER_PATTERNS = [
    r"^(?:pon|ponme|crea|activa|lanza)?\s*(?:un\s+)?temporizador\s+(?:de\s+)?(.+)",
    r"^av[ií]same\s+(?:en|dentro\s+de)\s+(.+)",
    r"^(?:en|dentro\s+de)\s+(\d+\s+(?:hora|horas|minuto|minutos|segundo|segundos))\s+av[ií]same",
]

TIMER_CANCEL_PATTERNS = [
    r"^(?:cancela|cancelar|para|parar|det[eé]n|detener|quita|quitar|borra|borrar|elimina|eliminar|anula|anular|olv[ií]date\s+del?)\s+(?:el\s+|los\s+|todos\s+los\s+|mi\s+)?temporizador(?:es)?\b",
    r"^temporizador(?:es)?\s+(?:cancela(?:do|dos)?|fuera|stop)\b",
]


def _parse_timer_duration(text: str) -> Optional[int]:
    """Parses a time expression and returns total seconds, or None if unparseable."""
    text = text.lower().strip().rstrip(".?!,¿¡")
    total = 0
    found = False

    patterns = [
        (r"(\d+)\s*hora(?:s)?", 3600),
        (r"media\s+hora", 1800),
        (r"cuarto\s+de\s+hora", 900),
        (r"(\d+)\s*minuto(?:s)?", 60),
        (r"medio\s+minuto", 30),
        (r"(\d+)\s*segundo(?:s)?", 1),
    ]

    for pattern, multiplier in patterns:
        m = re.search(pattern, text)
        if m:
            found = True
            value = int(m.group(1)) if m.lastindex else 1
            total += value * multiplier

    return total if found and total > 0 else None


MUSIC_PLAY_VERBS = r"(?:pon|ponme|ponte|reproduce|reproduceme|reprodúceme|escucha|escuchar|quiero\s+escuchar|quiero\s+oir|quiero\s+oír|dale\s+a|toca|tocame|tócame|busca|suena|sonar|cántame|cantame|canta)"
MUSIC_PLAY_FILLER = r"(?:la\s+canci[oó]n\s+|la\s+m[uú]sica\s+|el\s+tema\s+|m[uú]sica\s+de\s+|algo\s+de\s+|tema\s+de\s+)?"
MUSIC_PLAY_PATTERNS = [
    rf"^(?:por\s+favor\s+)?{MUSIC_PLAY_VERBS}\s+{MUSIC_PLAY_FILLER}(.+)",
]
MUSIC_STOP_PATTERNS = [
    r"\b(?:para|pausa|deten|detén|detente|c[aá]llate|silencio|stop|apaga|quita|cierra|corta)\s+(?:la\s+|el\s+)?(?:m[uú]sica|canci[oó]n|tema|reproducci[oó]n)\b",
    r"^(?:para|pausa|stop|c[aá]llate|silencio|basta)\.?$",
]

# --- Light control patterns --------------------------------------------------
LIGHT_ON_PATTERNS = [
    r"\b(?:enciende|encender|prende|prender|pon|conecta|activa)\b.{0,20}\bluc(?:es|e)?\b",
    r"\bluc(?:es|e)?\b.{0,15}\b(?:on|encendida|encendidas|encendido)\b",
    r"^(?:luz|luces)\s+on\.?$",
]
LIGHT_OFF_PATTERNS = [
    r"\b(?:apaga|apagar|quita|quitar|desconecta|desconectar|corta|cortar)\b.{0,20}\bluc(?:es|e)?\b",
    r"\bluc(?:es|e)?\b.{0,15}\b(?:off|apagada|apagadas|apagado)\b",
    r"^(?:luz|luces)\s+off\.?$",
]
LIGHT_TOGGLE_PATTERNS = [
    r"\b(?:cambia|toggle|cambia)\b.{0,15}\bluc(?:es|e)?\b",
]
LIGHT_STATUS_PATTERNS = [
    r"\b(?:estado|c[oó]mo\s+est[aá]|est[aá])\b.{0,20}\bluc(?:es|e)?\b",
    r"\bluc(?:es|e)?\b.{0,15}\b(?:encendida|apagada|estado)\b",
    r"^(?:luz|luces)\s+(?:estado|status)\??$",
]

# --- Voice triggers ----------------------------------------------------------
# Hard-coded "magic phrases" that bypass the LLM and play a specific song.
# Detection is substring-based on an accent-folded transcript.
#
# required_words is a list of GROUPS. Each group is a list of alternative
# spellings — at least ONE word from EACH group must appear in the
# transcript for the trigger to fire. This makes the matcher tolerant to
# Whisper mistranscriptions (e.g. "comendante" instead of "comandante").
#
# Each entry:
#   name:           log/debug label
#   required_words: list[list[str]] — every group needs one hit
#   music_query:    query passed to MusicPlayer.play()
#   announce:       optional TTS line spoken before playback (None to skip)
VOICE_TRIGGERS = [
    {
        "name": "El Comandante",
        "required_words": [
            # Group 1: "comandante" and common Whisper misfires
            ["comandante", "comendante", "comen dante", "comanda nte", "comandant"],
            # Group 2: "aqui" variants (already accent-folded)
            ["aqui", "aki", "aquim"],
        ],
        # Direct YouTube URL guarantees the exact version plays instead of
        # whatever yt-dlp's search would happen to return first.
        "music_query": "https://www.youtube.com/watch?v=r2DZzyeni6I",
        "announce": "A formar, el Comandante ha llegado.",
    },
]


def _normalize_for_trigger(text: str) -> str:
    """Lowercase + strip accents + collapse whitespace, for tolerant matching."""
    import unicodedata

    t = text.lower().strip().rstrip(".?!,¿¡")
    # Strip accents: "está" -> "esta", "aquí" -> "aqui"
    t = "".join(
        c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn"
    )
    return t


class VoiceAssistant:
    def __init__(self, args, client):
        self.args = args
        self.interrupt_event = threading.Event()
        self.conversation_count = 0
        self.is_handling_conversation = False

        # Active timers — each entry: {"event": Event, "label": str, "thread": Thread}
        self.active_timers = []
        self.timers_lock = threading.Lock()

        # Wake word detection improvements
        self.last_wakeword_time = 0
        self.wakeword_cooldown = 1.0  # Reduced from 1.5s
        self.consecutive_detection_count = 0
        self.required_consecutive = 2  # Confirmations needed

        logging.debug(
            f"VoiceAssistant init - cooldown: {self.wakeword_cooldown}s, required consecutive: {self.required_consecutive}"
        )

        # Initialize Subsystems
        self.audio = AudioInput(args)
        self.transcriber = Transcriber(args)
        self.tts = Synthesizer(args, self.interrupt_event)
        self.music = MusicPlayer()
        self.shelly = ShellyController(args)

        # Select LLM handler based on the resolved backend
        backend = getattr(args, "effective_llm_backend", "ollama")
        if backend == "groq":
            self.llm = GroqHandler(client, args)
            logging.info(f"LLM handler: Groq (model: {args.groq_model})")
        else:
            self.llm = LLMHandler(client, args)
            logging.info(f"LLM handler: Ollama (model: {args.ollama_model})")

        # Wakeword Setup
        if not os.path.exists(args.wakeword_model_path):
            raise FileNotFoundError(
                f"Wakeword model missing: {args.wakeword_model_path}"
            )

        logging.debug(f"Loading wakeword model from: {args.wakeword_model_path}")
        self.oww_model = Model(wakeword_model_paths=[args.wakeword_model_path])
        self.wakeword_key = list(self.oww_model.models.keys())[0]
        logging.debug(f"Wakeword model loaded with key: {self.wakeword_key}")

    def run(self):
        wakewords_display = "', '".join(self.args.wakewords_list)

        # Saludo de arranque sarcástico
        greeting = random.choice(STARTUP_GREETINGS)
        logging.info(f"Saludo: {greeting}")
        self.tts.speak(greeting)
        self.tts.queue.join()

        self.audio.start()
        logging.info(
            f"Activo en modo directo. Tras {DIRECT_MODE_TIMEOUT:.0f}s de silencio "
            f"se activará el wakeword '{wakewords_display}'."
        )

        last_interaction_time = time.time()
        in_wakeword_mode = False

        score_history = []
        weighted_scores = []

        try:
            while True:
                time_since_last = time.time() - last_interaction_time

                # Transición a modo wakeword si pasaron los 20s
                if not in_wakeword_mode and time_since_last >= DIRECT_MODE_TIMEOUT:
                    in_wakeword_mode = True
                    logging.info(
                        f"Silencio de {DIRECT_MODE_TIMEOUT:.0f}s alcanzado. "
                        f"Esperando wakeword '{wakewords_display}'..."
                    )
                    self.oww_model.reset()
                    score_history.clear()
                    weighted_scores.clear()
                    self.consecutive_detection_count = 0

                if not in_wakeword_mode:
                    # Modo directo: escucha comando sin wakeword ni "¿Sí?"
                    got_response = self._handle_conversation(skip_acknowledgment=True)
                    if got_response:
                        last_interaction_time = time.time()
                    continue

                # Modo wakeword
                chunk = self.audio.get_chunk()
                if not chunk:
                    time.sleep(0.001)
                    continue

                int16_audio = np.frombuffer(chunk, dtype=np.int16)
                prediction = self.oww_model.predict(int16_audio)
                score = prediction.get(self.wakeword_key, 0)

                score_history.append(score)
                if len(score_history) > 100:
                    score_history.pop(0)

                current_time = time.time()

                weighted_scores.append(score)
                if len(weighted_scores) > 5:
                    weighted_scores.pop(0)
                avg_score = sum(weighted_scores) / len(weighted_scores)

                if score > self.args.wakeword_threshold:
                    if current_time - self.last_wakeword_time > self.wakeword_cooldown:
                        self.consecutive_detection_count += 1
                        logging.debug(
                            f"Wakeword candidate (score: {score:.2f}, avg: {avg_score:.2f}, "
                            f"consecutive: {self.consecutive_detection_count}/{self.required_consecutive})"
                        )

                        if (
                            self.consecutive_detection_count
                            >= self.required_consecutive
                            and avg_score > self.args.wakeword_threshold * 0.85
                        ):
                            recent_scores = [f"{s:.2f}" for s in score_history[-10:]]
                            logging.info(
                                f"Wakeword detected! (score: {score:.2f}, avg: {avg_score:.2f}, "
                                f"recent: {', '.join(recent_scores)})"
                            )

                            self.last_wakeword_time = current_time
                            self.consecutive_detection_count = 0
                            weighted_scores.clear()
                            self.oww_model.reset()

                            # Confirmación con "¿Sí?" y conversación
                            self._handle_conversation(skip_acknowledgment=False)

                            # Volver a modo directo
                            last_interaction_time = time.time()
                            in_wakeword_mode = False
                            score_history.clear()
                            logging.info(
                                f"Modo directo reactivado. {DIRECT_MODE_TIMEOUT:.0f}s "
                                f"hasta volver al wakeword."
                            )
                    else:
                        time_since_wake = current_time - self.last_wakeword_time
                        logging.debug(
                            f"Wakeword in cooldown (score: {score:.2f}, "
                            f"since last: {time_since_wake:.2f}s)"
                        )
                else:
                    if self.consecutive_detection_count > 0:
                        logging.debug(f"Wakeword sequence broken (score: {score:.2f})")
                        self.consecutive_detection_count = 0

        except KeyboardInterrupt:
            logging.info("Stopping...")
        self.cleanup()

    def _handle_timer_command(self, text: str) -> bool:
        """Detects and sets/cancels a voice timer. Returns True if handled."""
        t = text.lower().strip().rstrip(".?!,¿¡")

        # --- Cancellation: handle BEFORE creation so "para el temporizador"
        # doesn't get parsed as a new "para...temporizador" creation. ---
        for pat in TIMER_CANCEL_PATTERNS:
            if re.match(pat, t, flags=re.IGNORECASE):
                with self.timers_lock:
                    pending = [t for t in self.active_timers if not t["event"].is_set()]
                    if not pending:
                        self.tts.speak(
                            "No hay ningún temporizador activo. Tranquilidad absoluta."
                        )
                        self.tts.queue.join()
                        return True
                    count = len(pending)
                    for timer in pending:
                        timer["event"].set()
                    self.active_timers.clear()

                if count == 1:
                    msg = "Temporizador cancelado. Como si nunca hubiera existido."
                else:
                    msg = f"Cancelados {count} temporizadores. Vaya lío llevabas."
                self.tts.speak(msg)
                self.tts.queue.join()
                logging.info(f"[Timer] Cancelados {count} temporizadores")
                return True

        # --- Creation ---
        for pat in TIMER_PATTERNS:
            m = re.match(pat, t, flags=re.IGNORECASE)
            if m:
                duration_text = m.group(1).strip()
                seconds = _parse_timer_duration(duration_text)
                if seconds is None:
                    self.tts.speak(
                        "No he entendido el tiempo. Dime algo como: temporizador de cinco minutos."
                    )
                    self.tts.queue.join()
                    return True

                # Format confirmation message
                if seconds >= 3600 and seconds % 3600 == 0:
                    label = (
                        f"{seconds // 3600} hora{'s' if seconds // 3600 != 1 else ''}"
                    )
                elif seconds >= 60 and seconds % 60 == 0:
                    label = f"{seconds // 60} minuto{'s' if seconds // 60 != 1 else ''}"
                else:
                    label = f"{seconds} segundo{'s' if seconds != 1 else ''}"

                self.tts.speak(
                    f"Temporizador de {label}. No se me olvidará, aunque me gustaría."
                )
                self.tts.queue.join()
                logging.info(f"[Timer] Iniciado: {seconds}s")

                cancel_event = threading.Event()

                def _fire(secs, lbl, tts, ev):
                    # Interruptible sleep: returns True if cancelled.
                    was_cancelled = ev.wait(timeout=secs)
                    if was_cancelled:
                        logging.info(f"[Timer] {lbl} cancelado antes de disparar")
                        return
                    logging.info(f"[Timer] Disparado tras {secs}s")
                    tts.speak(f"Han pasado {lbl}. Ya puede dejar de ignorarme.")
                    tts.queue.join()
                    # Self-clean from the active list once fired.
                    with self.timers_lock:
                        self.active_timers[:] = [
                            x for x in self.active_timers if x["event"] is not ev
                        ]

                t_thread = threading.Thread(
                    target=_fire,
                    args=(seconds, label, self.tts, cancel_event),
                    daemon=True,
                )
                with self.timers_lock:
                    self.active_timers.append(
                        {
                            "event": cancel_event,
                            "label": label,
                            "thread": t_thread,
                        }
                    )
                t_thread.start()
                return True

        return False

    def _handle_voice_triggers(self, text: str) -> bool:
        """
        Detects hard-coded voice triggers (e.g. 'el comandante está aquí')
        and plays a specific song via the music player. Bypasses the LLM.
        Uses substring matching on an accent-folded transcript to tolerate
        Whisper variants. Returns True if handled.
        """
        normalized = _normalize_for_trigger(text)
        logging.info(f"[VoiceTrigger] Evaluando: '{normalized}'")

        for trigger in VOICE_TRIGGERS:
            required = trigger["required_words"]
            # Each group must contribute at least one hit. Accept both
            # legacy flat list[str] and new list[list[str]] shapes.
            groups = [g if isinstance(g, (list, tuple)) else [g] for g in required]
            matched_per_group = [
                next((w for w in group if w in normalized), None) for group in groups
            ]
            if all(matched_per_group):
                name = trigger["name"]
                query = trigger["music_query"]
                logging.info(
                    f"[VoiceTrigger] '{name}' activado — hits {matched_per_group} "
                    f"en '{normalized}'"
                )

                if not self.music.available():
                    self.tts.speak("No tengo el reproductor de música disponible.")
                    self.tts.queue.join()
                    return True

                # Mirror the normal music command flow exactly:
                announce = trigger.get("announce")
                if announce:
                    self.tts.speak(announce)
                    self.tts.queue.join()
                else:
                    self.tts.speak(f"Buscando {query}.")
                    self.tts.queue.join()

                logging.info(f"[VoiceTrigger] Llamando music.play('{query}')")
                title = self.music.play(query)
                if title:
                    logging.info(f"[VoiceTrigger] '{name}' sonando: {title}")
                else:
                    logging.warning(
                        f"[VoiceTrigger] '{name}' — music.play devolvió None/vacío"
                    )
                    self.tts.speak("No he podido encontrarla.")
                    self.tts.queue.join()
                return True

        return False

    def _handle_music_command(self, text: str) -> bool:
        """Detects and executes music commands. Returns True if handled."""
        t = text.lower().strip().rstrip(".?!,¿¡")
        logging.info(f"[MUSIC] Evaluando: '{t}'")

        for pat in MUSIC_STOP_PATTERNS:
            if re.search(pat, t, flags=re.IGNORECASE):
                if self.music.stop():
                    logging.info("Música detenida por comando de voz.")
                    self.tts.speak("Hecho.")
                else:
                    self.tts.speak("No había nada sonando.")
                self.tts.queue.join()
                return True

        for pat in MUSIC_PLAY_PATTERNS:
            m = re.match(pat, t, flags=re.IGNORECASE)
            if m:
                query = m.group(m.lastindex).strip()
                if not query:
                    return False
                if not self.music.available():
                    self.tts.speak("No tengo el reproductor de música disponible.")
                    self.tts.queue.join()
                    return True
                logging.info(f"Comando música: '{query}'")
                self.tts.speak(f"Buscando {query}.")
                self.tts.queue.join()
                title = self.music.play(query)
                if title:
                    logging.info(f"Sonando: {title}")
                else:
                    self.tts.speak("No he podido encontrarla.")
                    self.tts.queue.join()
                return True

        return False

    def _handle_light_command(self, text: str) -> bool:
        """Detects and executes light control commands via Shelly. Returns True if handled."""
        if not self.shelly.available():
            return False

        t = text.lower().strip().rstrip(".?!,¿¡")
        logging.info(f"[LIGHT] Evaluando: '{t}'")

        for pat in LIGHT_ON_PATTERNS:
            if re.search(pat, t, flags=re.IGNORECASE):
                logging.info("[LIGHT] Comando encender detectado")
                if self.shelly.turn_on():
                    self.tts.speak("Luz encendida.")
                else:
                    self.tts.speak("No he podido contactar con el Shelly.")
                self.tts.queue.join()
                return True

        for pat in LIGHT_OFF_PATTERNS:
            if re.search(pat, t, flags=re.IGNORECASE):
                logging.info("[LIGHT] Comando apagar detectado")
                if self.shelly.turn_off():
                    self.tts.speak("Luz apagada.")
                else:
                    self.tts.speak("No he podido contactar con el Shelly.")
                self.tts.queue.join()
                return True

        for pat in LIGHT_TOGGLE_PATTERNS:
            if re.search(pat, t, flags=re.IGNORECASE):
                logging.info("[LIGHT] Comando toggle detectado")
                if self.shelly.toggle():
                    self.tts.speak("Hecho.")
                else:
                    self.tts.speak("No he podido contactar con el Shelly.")
                self.tts.queue.join()
                return True

        for pat in LIGHT_STATUS_PATTERNS:
            if re.search(pat, t, flags=re.IGNORECASE):
                logging.info("[LIGHT] Comando estado detectado")
                status = self.shelly.get_status()
                if status is None:
                    self.tts.speak("No he podido contactar con el Shelly.")
                elif status:
                    self.tts.speak("La luz está encendida.")
                else:
                    self.tts.speak("La luz está apagada.")
                self.tts.queue.join()
                return True

        return False

    def _process_plugins(self, text: str) -> str:
        """Processes simple plugins like [current time]."""
        if "[current time]" in text.lower():
            current_time = time.strftime("%I:%M %p")
            logging.debug(f"Plugin found: [current time] -> {current_time}")
            # Use regex for case-insensitive replacement
            text = re.sub(r"\[current time\]", current_time, text, flags=re.IGNORECASE)
        return text

    def _handle_conversation(self, skip_acknowledgment: bool = False) -> bool:
        """Returns True if the user spoke (any audio captured), False on silence/timeout."""
        try:
            conversation_start = time.time()

            # Optional memory profiling
            mem_before = 0
            if self.args.debug and self.args.memory_profiling:
                mem_before = monitor_memory()
                logging.debug(f"Memory at conversation start: {mem_before:.2f} MB")

            self.audio.stop()
            self.audio.clear_buffer()

            if not skip_acknowledgment:
                logging.debug("Playing acknowledgment")
                self.tts.speak("¿Sí?")
                self.tts.queue.join()

            self.interrupt_event.clear()

            # Start listening for command
            logging.debug("Starting audio recording for command")
            self.audio.start()

            # Longer delay to allow TTS audio to fade completely
            time.sleep(0.4)

            recording_start = time.time()
            audio_np = self.audio.record_phrase(
                self.interrupt_event, self.args.listen_timeout
            )
            recording_duration = time.time() - recording_start

            # Stop listening and process
            self.audio.stop()

            if audio_np is None:
                logging.debug(
                    f"No audio recorded (recording took {recording_duration:.2f}s)"
                )
                self.audio.start()
                return False

            logging.debug(f"Audio recording completed in {recording_duration:.2f}s")

            # IMPROVEMENT: More sophisticated audio quality validation
            audio_rms = np.sqrt(np.mean(audio_np**2))
            audio_peak = np.max(np.abs(audio_np))
            audio_std = np.std(audio_np)

            logging.debug(
                f"Audio quality - RMS: {audio_rms:.4f}, Peak: {audio_peak:.4f}, StdDev: {audio_std:.4f}"
            )

            # Check for multiple quality indicators
            if audio_rms < 0.01:
                logging.warning(
                    f"Audio too quiet (RMS: {audio_rms:.4f}), proceeding to transcription"
                )

            if audio_std < 0.005:
                logging.warning(
                    f"Audio lacks variation (StdDev: {audio_std:.4f}), likely silence, proceeding to transcription"
                )

            # Check if audio is clipping (saturated)
            if audio_peak > 0.98:
                logging.warning(f"Audio may be clipping (Peak: {audio_peak:.4f})")
                # Don't return - just warn, as clipped audio can still be transcribed

            # Transcribe with retry logic
            transcription_start = time.time()
            user_text = self._transcribe_with_retry(audio_np)
            transcription_duration = time.time() - transcription_start

            logging.debug(f"Transcription completed in {transcription_duration:.2f}s")

            # Explicitly release audio data from memory
            del audio_np

            if not user_text or not user_text.strip():
                logging.debug("Transcription was empty or whitespace only")
                self.audio.start()
                return True

            # Trim wake word if enabled
            original_text = user_text
            if self.args.trim_wake_word:
                user_text = self._trim_wakeword(user_text)
                if user_text != original_text:
                    logging.debug(
                        f"Wake word trimmed: '{original_text}' -> '{user_text}'"
                    )

            # If the command is now empty, do nothing
            if not user_text or not user_text.strip():
                logging.debug("Command empty after wake word trimming")
                self.audio.start()
                return True

            # Take only the first sentence
            sentences = re.split(r"(?<=[.?!])\s+", user_text)
            if sentences:
                first_sentence = sentences[0]
                if first_sentence != user_text:
                    logging.debug(f"Using first sentence only: '{first_sentence}'")
                    user_text = first_sentence

            # Process any plugins
            user_text = self._process_plugins(user_text)

            logging.info(f"You: {user_text}")

            # Check for exit commands (English and Spanish)
            user_text_lower = user_text.lower()
            if any(
                cmd in user_text_lower
                for cmd in ["exit", "goodbye", "salir", "adiós", "adios"]
            ):
                logging.debug("Exit command detected")
                self.tts.speak("Adiós.")
                self.tts.queue.join()
                exit(0)

            # Check for hard-coded voice triggers (e.g. "el comandante está aquí")
            # BEFORE timer/music/LLM so the magic phrase always wins.
            if self._handle_voice_triggers(user_text):
                self.audio.start()
                return True

            # Check for timer commands BEFORE sending to LLM
            if self._handle_timer_command(user_text):
                self.audio.start()
                return True

            # Check for music commands BEFORE sending to LLM
            if self._handle_music_command(user_text):
                self.audio.start()
                return True

            # Check for light control commands BEFORE sending to LLM
            if self._handle_light_command(user_text):
                self.audio.start()
                return True

            # Check for history reset commands (English and Spanish)
            if any(
                cmd in user_text_lower
                for cmd in [
                    "new chat",
                    "reset chat",
                    "nuevo chat",
                    "reiniciar chat",
                    "borrar historial",
                ]
            ):
                logging.debug("Chat reset command detected")
                self.llm.reset_history()
                self.tts.speak("Historial borrado.")
                self.tts.queue.join()
                self.audio.start()
                return True

            # Get LLM Response & Speak
            logging.debug("Sending to LLM")
            llm_start = time.time()
            sentence_buffer = ""
            token_count = 0

            for token in self.llm.chat_stream(user_text):
                if token is None:
                    logging.error("LLM returned None token")
                    break
                if self.interrupt_event.is_set():
                    logging.debug("Conversation interrupted")
                    self.tts.clear_queue()
                    break

                token_count += 1
                sentence_buffer += token

                # Stream sentences to TTS
                if any(p in token for p in SENTENCE_END_PUNCTUATION):
                    sentence = sentence_buffer.strip()
                    if sentence:
                        logging.debug(f"Queuing sentence for TTS: '{sentence[:50]}...'")
                        self.tts.speak(sentence)
                    sentence_buffer = ""

            llm_duration = time.time() - llm_start
            logging.debug(
                f"LLM streaming completed in {llm_duration:.2f}s ({token_count} tokens)"
            )

            # Speak remaining buffer
            if sentence_buffer.strip() and not self.interrupt_event.is_set():
                logging.debug(
                    f"Queuing final buffer for TTS: '{sentence_buffer.strip()}'"
                )
                self.tts.speak(sentence_buffer.strip())

            logging.debug("Waiting for TTS to complete")
            self.tts.queue.join()

            # After conversation completes
            self.conversation_count += 1
            conversation_duration = time.time() - conversation_start

            logging.debug(
                f"Conversation #{self.conversation_count} completed in {conversation_duration:.2f}s"
            )

            # Periodic aggressive cleanup
            if (
                self.args.gc_interval > 0
                and self.conversation_count % self.args.gc_interval == 0
            ):
                gc.collect()
                logging.debug(
                    f"Periodic garbage collection triggered (every {self.args.gc_interval} conversations)"
                )

            # Optional memory profiling
            if self.args.debug and self.args.memory_profiling and mem_before > 0:
                mem_after = monitor_memory()
                mem_delta = mem_after - mem_before
                logging.debug(
                    f"Memory at conversation end: {mem_after:.2f} MB (delta: {mem_delta:+.2f} MB)"
                )

            self.audio.start()
            return True
        finally:
            self.is_handling_conversation = False

    def _transcribe_with_retry(self, audio_np: np.ndarray, max_retries: int = 3) -> str:
        """Transcribe with progressive threshold relaxation and better logging."""
        original_logprob = self.args.whisper_avg_logprob
        original_nospeech = self.args.whisper_no_speech_prob

        # Define threshold progression
        threshold_steps = [
            (original_logprob, original_nospeech),
            (original_logprob - 0.15, original_nospeech + 0.1),
            (original_logprob - 0.3, original_nospeech + 0.2),
        ]

        logging.debug(
            f"Starting transcription (initial thresholds: logprob={original_logprob}, no_speech={original_nospeech})"
        )

        for attempt in range(min(max_retries, len(threshold_steps))):
            logprob_threshold, nospeech_threshold = threshold_steps[attempt]

            # Update thresholds
            self.args.whisper_avg_logprob = logprob_threshold
            self.args.whisper_no_speech_prob = nospeech_threshold

            logging.debug(
                f"Transcription attempt {attempt + 1}/{max_retries} (logprob={logprob_threshold:.2f}, no_speech={nospeech_threshold:.2f})"
            )

            user_text = self.transcriber.transcribe(audio_np)

            if user_text and user_text.strip():
                logging.debug(
                    f"Transcription successful on attempt {attempt + 1}: '{user_text}'"
                )
                # Restore original thresholds
                self.args.whisper_avg_logprob = original_logprob
                self.args.whisper_no_speech_prob = original_nospeech
                return user_text

            if attempt < max_retries - 1:
                logging.debug(
                    f"Attempt {attempt + 1} failed, trying with relaxed thresholds"
                )

        # Restore original thresholds
        self.args.whisper_avg_logprob = original_logprob
        self.args.whisper_no_speech_prob = original_nospeech

        logging.warning(f"All {max_retries} transcription attempts failed")
        return ""

    def _trim_wakeword(self, text: str) -> str:
        """Trims any configured wake word from the transcription using regex for robustness."""
        text_lower = text.lower().strip()

        # Build patterns from ALL configured wakewords
        patterns_to_match = []

        for wakeword in self.args.wakewords_list:
            wakeword_lower = wakeword.lower().strip()
            # Add the exact configured wake word
            patterns_to_match.append(re.escape(wakeword_lower))

            # Extract individual words to create flexible patterns
            words = wakeword_lower.split()
            if len(words) > 1:
                # Add pattern with optional prefix (e.g., "hey" from "hey jarvis")
                # Treat first word as optional prefix
                core_name = words[-1]
                prefix = r"\s*".join(re.escape(w) for w in words[:-1])
                patterns_to_match.append(
                    r"(?:" + prefix + r"\s*)?" + re.escape(core_name)
                )
            else:
                patterns_to_match.append(re.escape(wakeword_lower))

        # Generate common misspellings/pronunciations for known wakeword names
        core_wakeword_names = [
            "jarvis",
            "jarlis",
            "jarvas",
            "jarves",
            "jarvys",
            "jarvois",
        ]
        for name in core_wakeword_names:
            patterns_to_match.append(r"(?:hey\s*)?" + re.escape(name))
            patterns_to_match.append(re.escape(name))

        # Create a single regex pattern to match any of these at the start or end,
        # with optional punctuation and spaces. Use word boundaries where appropriate.
        # This regex will look for the pattern either at the beginning (^) or the end ($)
        # of the string, allowing for flexible matching.

        # Example: if wakeword is "hey jarvis"
        # patterns_to_match could be: ["hey\\s*jarvis", "hey\\s*jarlis", ..., "jarvis", "jarlis", ...]

        # Construct the full regex:
        # 1. Match at the beginning: (?:<pattern>)\b[.,!?]*\s*
        # 2. Match at the end: \s*\b(?:<pattern>)[.,!?]*$

        # To avoid over-trimming, ensure word boundaries (\b) are used where logical.
        # Also, make sure the most specific patterns are tried first if using an OR separated list.

        # For simplicity and to avoid complex lookarounds, we'll try to find the longest match first
        # and then trim. A single regex can be structured to capture the matched wake word part.

        # Let's build a regex that captures the wake word part we want to remove.
        # We need to ensure we don't accidentally trim valid speech that happens to contain
        # a wake word component.

        # Pattern for matching at the beginning (case-insensitive)
        # e.g., "hey jarvis, what time" -> "what time"
        # (?:^|\s) ensures we match at the start or after a space, \b for word boundary
        combined_start_pattern = (
            r"^(?:" + "|".join(patterns_to_match) + r")\b[.,!?]*\s*"
        )
        match_start = re.match(combined_start_pattern, text_lower, re.IGNORECASE)
        if match_start:
            trimmed_text = text[len(match_start.group(0)) :].strip()
            logging.debug(
                f"Wake word trimmed from start: '{match_start.group(0)}' removed. Result: '{trimmed_text}'"
            )
            return trimmed_text

        # Pattern for matching at the end (case-insensitive)
        # e.g., "what time hey jarvis" -> "what time"
        combined_end_pattern = r"\s*\b(?:" + "|".join(patterns_to_match) + r")[.,!?]*$"
        match_end = re.search(combined_end_pattern, text_lower, re.IGNORECASE)
        if match_end:
            trimmed_text = text[: match_end.start()].strip()
            logging.debug(
                f"Wake word trimmed from end: '{match_end.group(0)}' removed. Result: '{trimmed_text}'"
            )
            return trimmed_text

        logging.debug("No wake word pattern found, keeping original text")
        return text

    def cleanup(self):
        logging.debug("Starting cleanup")
        self.music.stop()
        self.audio.stop()
        self.tts.stop()
        self.transcriber.close()
        logging.debug("Cleanup complete")
