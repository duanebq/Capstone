from faster_whisper import WhisperModel

print("Loading model…")
model = WhisperModel("tiny", compute_type="int8")
segments, info = model.transcribe("test_audio.wav", beam_size=1)

print("Language:", info.language, "Prob:", round(info.language_probability, 3))
for s in segments:
    print(f"[{s.start:.2f} -> {s.end:.2f}] {s.text}")
