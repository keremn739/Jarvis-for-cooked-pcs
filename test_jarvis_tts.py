import asyncio
import edge_tts

VOICE = "en-US-AndrewNeural"

lines = [
    "Hello. I'm Jarvis. How can I help?",
    "Sure. I'll take care of that.",
    "I've opened Chrome for you.",
    "I found three results. The first one looks like the most relevant.",
    "Your CPU usage is currently at twenty-three percent.",
    "I couldn't complete that request. The application isn't responding.",
    "Good morning. You have two classes today.",
    "Alright, I'm listening.",
    "I didn't quite catch that. Could you repeat it?",
]

async def main():
    for i, text in enumerate(lines, 1):
        print(f"[{i}] {text}")
        communicate = edge_tts.Communicate(text, VOICE)
        await communicate.save(f"/tmp/jarvis_tts_{i}.mp3")

asyncio.run(main())
