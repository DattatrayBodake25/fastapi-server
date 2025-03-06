from fastapi import FastAPI, HTTPException
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
    try:
        # Validate input texts
        if not data.text1 or not data.text2:
            raise HTTPException(status_code=400, detail="Both 'text1' and 'text2' must be non-empty.")

        # Compute similarity score
        score = compute_similarity(data.text1, data.text2)

        # Return response
        return {"similarity_score": score}

    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Add root endpoint for health check
@app.get("/")
def health_check():
    return {"status": "API is running"}

# Add docs endpoint for API documentation
@app.get("/docs")
def api_docs():
    return {"message": "Visit /docs for Swagger UI documentation."}