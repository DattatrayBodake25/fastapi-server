from fastapi import FastAPI
from pydantic import BaseModel
from similarity_model import compute_similarity  # Import function from similarity_model.py
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load the processed dataset (adjust the path accordingly)
dataset = pd.read_csv('similarity_results.csv')

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
    # Calculate similarity score using the model
    score = compute_similarity(data.text1, data.text2)

    # Check if the pair exists in the dataset (optional step for testing)
    matching_rows = dataset[
        (dataset['text1'] == data.text1) & (dataset['text2'] == data.text2)
    ]
    
    if not matching_rows.empty:
        return {"similarity_score": score, "message": "Query matches existing dataset"}
    else:
        return {"similarity_score": score, "message": "No direct match found in dataset"}








# from fastapi import FastAPI
# from pydantic import BaseModel
# from similarity_model import compute_similarity  # Import function from similarity_model.py

# # Initialize FastAPI app
# app = FastAPI()

# # Define request body structure
# class TextPair(BaseModel):
#     text1: str
#     text2: str

# @app.post("/predict")
# def predict_similarity(data: TextPair):
#     """
#     API endpoint to compute similarity between two texts.
#     Request body:
#     {
#         "text1": "Your first paragraph here...",
#         "text2": "Your second paragraph here..."
#     }
#     Response:
#     {
#         "similarity_score": 0.8
#     }
#     """
#     score = compute_similarity(data.text1, data.text2)
#     return {"similarity_score": score}
