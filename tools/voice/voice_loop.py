import asyncio
import os
import time
import wave
import subprocess

import edge_tts
import sounddevice as sd
from scipy.signal import resample_poly
from groq import Groq
from silero_vad import load_silero_vad, VADIterator


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

INPUT_DEVICE = 13
INPUT_RATE = 48000
VAD_RATE = 16000
CHANNELS = 1

CHUNK_SAMPLES = 1536
VAD_CHUNK_SAMPLES = 512

VAD_THRESHOLD = 0.5
MIN_SILENCE_MS = 700
SPEECH_PAD_MS = 30
PRE_ROLL_MS = 300

STT_MODEL = "whisper-large-v3-turbo"
TTS_VOICE = "en-US-AndrewNeural"

WAV_PATH = "/tmp/jarvis_voice_loop.wav"
TTS_PATH = "/tmp/jarvis_response.mp3"


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def save_wav(audio):
    audio_int16 = (
        audio * 32767
    ).clip(-32768, 32767).astype("int16")

    with wave.open(WAV_PATH, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(VAD_RATE)
        wav.writeframes(audio_int16.tobytes())


async def speak(text):
    communicate = edge_tts.Communicate(
        text,
        TTS_VOICE,
    )

    await communicate.save(TTS_PATH)

    subprocess.run(
        [
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "quiet",
            TTS_PATH,
        ],
        check=False,
    )


# ─────────────────────────────────────────────
# VAD
# ─────────────────────────────────────────────

print("Loading Silero VAD...")

vad_model = load_silero_vad(onnx=True)

vad = VADIterator(
    vad_model,
    threshold=VAD_THRESHOLD,
    sampling_rate=VAD_RATE,
    min_silence_duration_ms=MIN_SILENCE_MS,
    speech_pad_ms=SPEECH_PAD_MS,
)

print("VAD ready.")


# ─────────────────────────────────────────────
# Groq
# ─────────────────────────────────────────────

if not os.environ.get("GROQ_API_KEY"):
    raise RuntimeError(
        "GROQ_API_KEY is not set in this terminal."
    )

groq_client = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)


# ─────────────────────────────────────────────
# Listen for one utterance
# ─────────────────────────────────────────────

def listen():
    print()
    print("Listening...")
    print("(Speak normally. Stop talking when finished.)")

    speech_audio = []
    recording = False
    finished = False

    pre_roll_samples = int(
        PRE_ROLL_MS * VAD_RATE / 1000
    )

    pre_roll = []

    def callback(indata, frames, time_info, status):
        nonlocal recording, finished

        if status:
            print("Audio status:", status)

        audio_48k = indata[:, 0].copy()

        audio_16k = resample_poly(
            audio_48k,
            VAD_RATE,
            INPUT_RATE,
        )

        # Keep a small rolling buffer before speech starts.
        pre_roll.append(audio_16k)

        total = sum(len(x) for x in pre_roll)

        while total > pre_roll_samples:
            pre_roll.pop(0)
            total = sum(len(x) for x in pre_roll)

        # Feed Silero in exactly 512-sample chunks.
        for start in range(
            0,
            len(audio_16k),
            VAD_CHUNK_SAMPLES,
        ):
            chunk = audio_16k[
                start:start + VAD_CHUNK_SAMPLES
            ]

            if len(chunk) < VAD_CHUNK_SAMPLES:
                break

            result = vad(chunk)

            if result and "start" in result:
                if not recording:
                    print(">>> SPEECH START")

                    recording = True

                    speech_audio.extend(
                        pre_roll
                    )

            if recording:
                speech_audio.append(chunk)

            if result and "end" in result:
                print("<<< SPEECH END")
                finished = True
                raise sd.CallbackStop()

    with sd.InputStream(
        samplerate=INPUT_RATE,
        channels=CHANNELS,
        dtype="float32",
        device=INPUT_DEVICE,
        blocksize=CHUNK_SAMPLES,
        callback=callback,
    ):
        while not finished:
            sd.sleep(50)

    if not speech_audio:
        return None

    return (
        __import__("numpy")
        .concatenate(speech_audio)
    )


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

async def main():

    while True:

        try:
            audio = listen()

        except KeyboardInterrupt:
            print("\nStopping.")
            break

        if audio is None:
            continue

        duration = len(audio) / VAD_RATE

        print()
        print(
            f"Captured audio: {duration:.2f} seconds"
        )

        save_wav(audio)

        print("Sending speech to Groq...")

        start = time.perf_counter()

        with open(WAV_PATH, "rb") as audio_file:
            transcription = (
                groq_client
                .audio
                .transcriptions
                .create(
                    file=audio_file,
                    model=STT_MODEL,
                )
            )

        stt_latency = time.perf_counter() - start

        text = transcription.text.strip()

        print()
        print("YOU:")
        print(text)

        print()
        print(
            f"Groq latency: {stt_latency:.2f}s"
        )

        if not text:
            continue

        print()
        print("JARVIS:")
        print("Repeating your request...")

        await speak(text)

        print()
        print("Ready for next request.")


if __name__ == "__main__":
    asyncio.run(main())
