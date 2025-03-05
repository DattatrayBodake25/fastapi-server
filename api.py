from fastapi import FastAPI
from pydantic import BaseModel
from similarity_model import compute_similarity  # Import function from similarity_model.py

# Initialize FastAPI app
app = FastAPI()

# Define request body structure
class TextPair(BaseModel):
    text1: str
    text2: str

@app.post("/predict")
def predict_similarity(data: TextPair):
    """
    API endpoint to compute similarity between two texts.
    Request body:
    {
        "text1": "Your first paragraph here...",
        "text2": "Your second paragraph here..."
    }
    Response:
    {
        "similarity_score": 0.8
    }
    """
    score = compute_similarity(data.text1, data.text2)
    return {"similarity_score": score}