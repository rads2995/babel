import whisper

model = whisper.load_model("turbo", download_root=".")
result = model.transcribe("test.mp3")
print(result["text"])
