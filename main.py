# main.py
from fastapi import FastAPI, UploadFile, File
import whisper
import tempfile

model = whisper.load_model("tiny")

app = FastAPI()

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    result = model.transcribe(tmp_path, language="fr")
    return {"text": result["text"]}
