# main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import whisper
import tempfile
from rapidfuzz import fuzz

# Load the small multilingual model
model = whisper.load_model("tiny")

app = FastAPI()

# CORS: only allow your frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # replace with production domain later
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

    # Use fuzzy matching to be more tolerant of punctuation/small differences
    score = fuzz.token_set_ratio(actual, expected)
    match = score >= 70  # or lower to be even more forgiving

    return {
        "actual": actual,
        "match": match,
        "score": score,
    }
