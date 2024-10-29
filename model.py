
#!/home/physicien/myenv/bin/python

import torch
import torch.nn as nn

from preprocess import model_wiki


class TowerQ(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(TowerQ, self).__init__()
        embedding_tensor = torch.tensor(model_wiki.vectors, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(embedding_tensor, freeze=False)  # Fine-tuning Word2Vec embeddings
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        _, x = self.rnn(x)  # x is the final hidden state
        return x.squeeze(0)

class TowerD(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(TowerD, self).__init__()
        embedding_tensor = torch.tensor(model_wiki.vectors, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(embedding_tensor, freeze=False)  # Fine-tuning Word2Vec embeddings
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        _, x = self.rnn(x)  # x is the final hidden state
        return x.squeeze(0)

# distance function for loss function
def cosine_distance(tensor1, tensor2):

    dot_product = torch.sum(tensor1 * tensor2, dim=1)
    norm1 = torch.norm(tensor1, dim = 1)
    norm2 = torch.norm(tensor2,dim = 1)

    cosine_distance = 1 - dot_product / (norm1 * norm2 + 1e-8)

    return cosine_distance

# create the loss function: L(q,doc+,doc-,d, m) = max(0,d(q,doc+) - d(q,doc-) + m)
def triplet_loss_function(query, relevant_document, irrelevant_document, margin):
    relevant_distance = cosine_distance(query, relevant_document)
    irrelevant_distance = cosine_distance(query, irrelevant_document)
    triplet_loss = torch.relu(relevant_distance - irrelevant_distance + margin)
    return triplet_loss.mean()
