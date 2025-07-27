# main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import whisper
import tempfile
from rapidfuzz import fuzz

# Load the multilingual tiny model once (NOT tiny.en)
model = whisper.load_model("tiny")

app = FastAPI()

# Restrict CORS to your frontend (localhost during dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # change to prod URL if needed
    allow_credentials=True,
    allow_methods=["POST", "GET"],
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

    # Fuzzy matching (more forgiving)
    score = fuzz.token_set_ratio(actual, expected)

    if len(expected.split()) <= 2:
        match = score >= 60  # more forgiving for short phrases
    else:
        match = score >= 70

    # Log for debugging (will show in Render logs)
    print("EXPECTED:", expected)
    print("ACTUAL:", actual)
    print("SCORE:", score)

    return {
        "actual": actual,
        "match": match,
        "score": score,
    }

# Health check endpoint to keep Render instance warm
@app.get("/ping")
def ping():
    return {"status": "ok"}
