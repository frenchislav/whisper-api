# main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import whisper
import tempfile

# Load smaller multilingual model
model = whisper.load_model("tiny")  # NOT "tiny.en"

app = FastAPI()

# ✅ Restrict CORS to your local frontend or production domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your deployed domain here later
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), expected: str = Form("")):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    result = model.transcribe(tmp_path, language="fr", task="transcribe")
    actual = result["text"].strip().lower()
    expected = expected.strip().lower()
    match = actual == expected
    score = round(100 * (1 - min(len(set(actual.split()) ^ set(expected.split())) / max(len(actual.split()), 1), 1)))

    return {
        "actual": actual,
        "match": match,
        "score": score,
    }
