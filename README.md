# Semantic Text Similarity API 🚀

This project implements a **Semantic Textual Similarity (STS) API** using **Sentence-BERT (SBERT)** to evaluate the similarity between pairs of text paragraphs. The API is built with **FastAPI** and deployed on **Render**, enabling real-time text similarity predictions.

## 📌 Features
- Computes **semantic similarity** between two paragraphs.
- Uses **SBERT (all-MiniLM-L6-v2)** for sentence embeddings.
- Fast and efficient **cosine similarity** computation.
- **FastAPI** for high-performance API development.
- **Render deployment** for automatic cloud hosting.


## 🚀 API Endpoints
### 1️⃣ Check API Status  
**`GET /`**  
Returns a welcome message.

### 2️⃣ Get Similarity Score  
**`POST /predict`**  
#### 📥 Request Format (JSON):
```json
{
  "text1": "First paragraph content...",
  "text2": "Second paragraph content..."
}
```

📤 Response Format (JSON):
```json
{
  "similarity_score": 0.73
}
```

## 🛠️ Setup & Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/DattatrayBodake25/fastapi-server.git
cd fastapi-server
```
### 2️⃣ Create a Virtual Environment (Optional)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Run the API Locally
```bash
uvicorn main:app --reload
```
The API will be available at: http://127.0.0.1:8000

## ☁️ Deployment on Render
The API is automatically deployed on Render. Whenever changes are pushed to the GitHub repository, Render updates the deployment.

🔗 Live API Endpoint: https://fastapi-server-6uc6.onrender.com
🔗 Swagger Docs: https://fastapi-server-6uc6.onrender.com/docs
