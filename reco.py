#!/home/physicien/myenv/bin/python

import torch
from model import TowerQ, TowerD  
from preprocess import tokenizer, triplets
import pickle 

# Load the model weights
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Instantiate new model instances
tower_q = TowerQ(embedding_dim = 50, hidden_dim = 100).to(device)  # Ensure parameters match training
tower_d = TowerD(embedding_dim = 50, hidden_dim = 100).to(device)  # Ensure parameters match training

# Download the weights from WandB
wandb_path_query = 'model_tower_query_final.pth'  # Replace <run_id> with your actual run ID
wandb_path_document = 'model_tower_document_final.pth'  # Replace <run_id> with your actual run ID

# Load the pretrained weights
tower_q.load_state_dict(torch.load(wandb_path_query, map_location=device))
tower_d.load_state_dict(torch.load(wandb_path_document, map_location=device))

# Set the models to evaluation mode
tower_q.eval()
tower_d.eval()

# Load document embeddings (if you saved them separately, otherwise recreate them)
try:
    with open('document_embeddings.pkl', 'rb') as f:
        document_embeddings = pickle.load(f)
except FileNotFoundError:
    document_embeddings = {}
    for doc_id, doc_tokens in enumerate(triplets['docpos']):
        doc_hidden_state = tower_d(torch.tensor(doc_tokens).unsqueeze(0).to(device))
        document_embeddings[doc_id] = doc_hidden_state
    with open('document_embeddings.pkl', 'wb') as f:
        pickle.dump(document_embeddings, f)


    # Get the hidden state for the query from the tower model
def recommend_documents(query, top_k=5):
    query_tokens = tokenizer(query)
    query_hidden = tower_q(torch.tensor(query_tokens).unsqueeze(0).to(device))

    similarities = [(doc_id, torch.nn.functional.cosine_similarity(query_hidden, doc_emb).item())
                    for doc_id, doc_emb in document_embeddings.items()]
    
    top_docs = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    return [doc_id for doc_id, _ in top_docs]

# Test the recommendation function
if __name__ == "__main__":
    query = "who is not a good journalist?"  # Replace with your test query
    top_recommendations = recommend_documents(query, top_k=5)
    print("Top document recommendations:", top_recommendations)


