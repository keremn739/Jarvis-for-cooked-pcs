import os
import time
import wave
from collections import deque

import numpy as np
import sounddevice as sd
import torch
from scipy.signal import resample_poly
from groq import Groq
from silero_vad import load_silero_vad, VADIterator


# ============================================================
# Configuration
# ============================================================

INPUT_DEVICE = 13          # Logitech C270
INPUT_RATE = 48000         # C270 native rate
VAD_RATE = 16000           # Silero VAD rate
CHANNELS = 1

CHUNK_SAMPLES = 1536       # 1536 @ 48 kHz = 512 @ 16 kHz
VAD_CHUNK_SAMPLES = 512

VAD_THRESHOLD = 0.5
MIN_SILENCE_MS = 700
SPEECH_PAD_MS = 30

PRE_ROLL_MS = 300

WAV_PATH = "/tmp/jarvis_voice_pipeline.wav"

MODEL_NAME = "whisper-large-v3-turbo"


# ============================================================
# Helpers
# ============================================================

def save_wav(path, audio_16k):
    """Save mono float32 audio as 16-bit PCM WAV."""
    audio_int16 = (
        audio_16k * 32767
    ).clip(-32768, 32767).astype(np.int16)

    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(VAD_RATE)
        wav.writeframes(audio_int16.tobytes())


# ============================================================
# Load local VAD
# ============================================================

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
print()
print("Waiting for speech...")
print("(Speak normally. The test ends after one utterance.)")
print()


# ============================================================
# Recording
# ============================================================

speech_audio = []
pre_roll = deque(
    maxlen=max(1, int(PRE_ROLL_MS / 1000 * INPUT_RATE / CHUNK_SAMPLES))
)

speech_started = False
speech_start_time = None
speech_end_time = None


def audio_callback(indata, frames, time_info, status):
    global speech_started
    global speech_start_time
    global speech_end_time

    if status:
        print(f"Audio status: {status}")

    # C270: 48 kHz float32
    chunk_48k = indata[:, 0].copy()

    # 48 kHz -> 16 kHz
    chunk_16k = resample_poly(
        chunk_48k,
        VAD_RATE,
        INPUT_RATE,
    ).astype(np.float32)

    # Keep a small amount of audio before speech starts.
    pre_roll.append(chunk_16k.copy())

    # Silero expects a 512-sample chunk at 16 kHz.
    if len(chunk_16k) != VAD_CHUNK_SAMPLES:
        return

    vad_result = vad(torch.from_numpy(chunk_16k))

    if vad_result is not None:

        # Speech started
        if "start" in vad_result and not speech_started:
            speech_started = True
            speech_start_time = time.perf_counter()

            print(">>> SPEECH START")

            # Include audio immediately before detection.
            speech_audio.extend(list(pre_roll))

        # Speech ended
        elif "end" in vad_result and speech_started:
            speech_started = False
            speech_end_time = time.perf_counter()

            print("<<< SPEECH END")

            # Include the final chunk.
            speech_audio.append(chunk_16k.copy())


    # Keep collecting while speech is active.
    if speech_started:
        speech_audio.append(chunk_16k.copy())


# ============================================================
# Start microphone
# ============================================================

with sd.InputStream(
    device=INPUT_DEVICE,
    samplerate=INPUT_RATE,
    channels=CHANNELS,
    dtype="float32",
    blocksize=CHUNK_SAMPLES,
    callback=audio_callback,
):
    while speech_end_time is None:
        time.sleep(0.01)


# ============================================================
# Build speech segment
# ============================================================

if not speech_audio:
    raise RuntimeError("No speech audio was captured.")

audio = np.concatenate(speech_audio)

# Remove excessive accidental length if necessary.
audio = audio.astype(np.float32)

save_wav(WAV_PATH, audio)

print()
print(f"Captured audio: {len(audio) / VAD_RATE:.2f} seconds")
print(f"Saved: {WAV_PATH}")


# ============================================================
# Send to Groq
# ============================================================

print()
print("Sending speech segment to Groq...")

client = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)

api_start = time.perf_counter()

with open(WAV_PATH, "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        file=audio_file,
        model=MODEL_NAME,
    )

api_latency = time.perf_counter() - api_start


# ============================================================
# Results
# ============================================================

print()
print("=" * 50)
print("TRANSCRIPT:")
print(transcription.text)
print("=" * 50)

print()
print(f"Groq API latency:       {api_latency:.2f} s")

if speech_end_time is not None and speech_start_time is not None:
    print(
        f"Speech detection time:  "
        f"{speech_end_time - speech_start_time:.2f} s"
    )

print()
print("Pipeline complete.")
