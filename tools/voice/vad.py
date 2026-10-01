import sounddevice as sd
import numpy as np
from scipy.signal import resample_poly
from silero_vad import load_silero_vad, VADIterator


# --- Configuration ---
MIC_DEVICE = 13
INPUT_RATE = 48000
VAD_RATE = 16000
CHUNK = 1536

VAD_THRESHOLD = 0.5
MIN_SILENCE_MS = 700
SPEECH_PAD_MS = 30

TEST_DURATION_SECONDS = 15


# --- Load local Silero ONNX model ---
print("Loading Silero VAD...")
model = load_silero_vad(onnx=True)

vad = VADIterator(
    model,
    threshold=VAD_THRESHOLD,
    sampling_rate=VAD_RATE,
    min_silence_duration_ms=MIN_SILENCE_MS,
    speech_pad_ms=SPEECH_PAD_MS,
)

print()
print(f"Microphone: device {MIC_DEVICE}")
print(f"Minimum silence: {MIN_SILENCE_MS} ms")
print(f"Test duration: {TEST_DURATION_SECONDS} seconds")
print()
print("Listening...")
print("Try:")
print('  "Merhaba Jarvis" -> short pause -> "what is the time?"')
print("  then a longer pause and another sentence.")
print()


def callback(indata, frames, time, status):
    if status:
        print("Audio:", status)

    audio48 = indata[:, 0].copy()

    # C270: 48 kHz -> Silero: 16 kHz
    audio16 = resample_poly(
        audio48,
        1,
        3,
    ).astype(np.float32)

    result = vad(audio16)

    if result is not None:
        if "start" in result:
            print(">>> SPEECH START")

        elif "end" in result:
            print("<<< SPEECH END")


with sd.InputStream(
    samplerate=INPUT_RATE,
    channels=1,
    dtype="float32",
    device=MIC_DEVICE,
    blocksize=CHUNK,
    callback=callback,
):
    sd.sleep(TEST_DURATION_SECONDS * 1000)


print()
print("Test finished.")
