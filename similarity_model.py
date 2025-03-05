import torch
import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load the pre-trained SBERT model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def preprocess_text(text):
    """
    Preprocesses text by:
    - Removing HTML tags
    - Converting text to lowercase
    - Stripping extra spaces

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    if not text or not isinstance(text, str):
        return ""  # Handle empty or invalid inputs

    text = text.lower().strip()  # Normalize case and remove extra spaces
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'\s+', ' ', text)  # Normalize multiple spaces
    return text

def compute_similarity(text1, text2):
    """
    Computes the cosine similarity score between two texts using SBERT embeddings.

    Args:
        text1 (str): First text.
        text2 (str): Second text.

    Returns:
        float: Similarity score between 0 and 1.
    """
    text1_clean = preprocess_text(text1)
    text2_clean = preprocess_text(text2)

    # Generate SBERT embeddings
    embedding1 = model.encode(text1_clean, convert_to_tensor=True)
    embedding2 = model.encode(text2_clean, convert_to_tensor=True)

    # Compute cosine similarity
    similarity_score = util.pytorch_cos_sim(embedding1, embedding2).item()

    # Ensure similarity is in the range [0,1]
    return round(max(0, similarity_score), 1)

def process_dataset(file_path):
    """
    Loads a dataset, computes similarity scores, and saves the results.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        None
    """
    df = pd.read_csv(file_path)

    # Validate dataset structure
    if "text1" not in df.columns or "text2" not in df.columns:
        raise ValueError("Dataset must contain 'text1' and 'text2' columns.")

    # Compute similarity scores for each text pair
    df["similarity_score"] = df.apply(lambda row: compute_similarity(row["text1"], row["text2"]), axis=1)

    # Save results
    output_file = "similarity_results.csv"
    df.to_csv(output_file, index=False)
    print(f"Similarity scores saved to: {output_file}")

if __name__ == "__main__":
    dataset_path = "DataNeuron_Text_Similarity.csv"
    process_dataset(dataset_path)