import os
import time
import wave

import sounddevice as sd
from scipy.signal import resample_poly
from groq import Groq


INPUT_DEVICE = 13
INPUT_RATE = 48000
STT_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 5

WAV_PATH = "/tmp/jarvis_groq_test.wav"


print("Recording...")

audio = sd.rec(
    int(RECORD_SECONDS * INPUT_RATE),
    samplerate=INPUT_RATE,
    channels=CHANNELS,
    dtype="float32",
    device=INPUT_DEVICE,
)

sd.wait()

print("Recording finished.")


# Convert C270 audio: 48 kHz -> 16 kHz
audio_16k = resample_poly(
    audio[:, 0],
    STT_RATE,
    INPUT_RATE,
)


# Convert float32 [-1, 1] -> signed 16-bit PCM
audio_int16 = (
    audio_16k * 32767
).clip(-32768, 32767).astype("int16")


# Save WAV
with wave.open(WAV_PATH, "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(STT_RATE)
    wav.writeframes(audio_int16.tobytes())


print("Sending audio to Groq...")


# API key comes from the environment.
# We never put the key directly in this file.
client = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)


start = time.perf_counter()


with open(WAV_PATH, "rb") as audio_file:

    transcription = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-large-v3-turbo",
    )


elapsed = time.perf_counter() - start


print()
print("TRANSCRIPT:")
print(transcription.text)

print()
print(f"API latency: {elapsed:.2f} seconds")
