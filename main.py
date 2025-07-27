# main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import whisper
import tempfile
from rapidfuzz import fuzz

# Load tiny multilingual model once (do NOT use "tiny.en")
model = whisper.load_model("tiny")

app = FastAPI()

# Enable CORS for local frontend only (secure)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # replace with prod domain later
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Whisper transcription endpoint
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), expected: str = Form("")):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Force French language
    result = model.transcribe(tmp_path, language="fr", task="transcribe")

    actual = result["text"].strip().lower()
    expected = expected.strip().lower()

    # Fuzzy match score using token set ratio
    score = fuzz.token_set_ratio(actual, expected)

    # More forgiving for short words
    if len(expected.split()) <= 2:
        match = score >= 60
    else:
        match = score >= 70

    print("EXPECTED:", expected)
    print("ACTUAL:", actual)
    print("SCORE:", score)

    return {
        "actual": actual,
        "match": match,
