# retrieve top 5 documents for new query

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch

# Load your recommendation models and preprocess function (pre-trained models should already be available)
tower_query = None  # This will hold the loaded query tower model
document_embeddings = {}  # This will hold the document embeddings

# Define FastAPI app
app = FastAPI()

# Data model for incoming request
class QueryRequest(BaseModel):
    query: str


# API to get top relevant documents
@app.post("/recommend")
def get_recommendations(request: QueryRequest):
    query = request.query
    if not query:
        raise HTTPException(status_code=400, detail="Query is empty.")
    
    top_docs = recommend_documents(query, tower_query, document_embeddings, preprocess_function, model_wiki, device)
    return {"query": query, "top_documents": top_docs}

# Load model and embeddings on startup
@app.on_event("startup")
def load_model_and_embeddings():
    global tower_query, document_embeddings, preprocess_function, model_wiki, device
    
    # Initialize or load pre-trained tower query model and document embeddings here
    tower_query, document_embeddings = main_pipeline(
        file_path='path_to_parquet_file',
        model_wiki='glove-wiki-gigaword-50',  # Loaded pre-trained model
        preprocess_function=preprocess,  # Preprocess function
        input_size=50,  # Embedding size
        hidden_size=128
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Health check API
@app.get("/health")
def health_check():
    return {"status": "up"}