import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel


MODEL = WhisperModel(
    "large-v3-turbo",
    device="cpu",
    compute_type="int8"
)


def listen(duration=5):
    print("Listening...")

    audio = sd.rec(
        int(duration * 16000),
        samplerate=16000,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    audio = np.squeeze(audio)

    segments, info = MODEL.transcribe(
        audio,
        language=None,
        beam_size=5,
        vad_filter=True,
        initial_prompt="Jarvis, JARVIS"
    )

    text = ""

    for segment in segments:
        text += segment.text

    print(f"Detected language: {info.language}")

    return text.strip()