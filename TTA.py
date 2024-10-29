## lazy one single script

#!/home/physicien/myenv/bin/python

import pandas as pd
import polars as pl
import random
import fastparquet
import pyarrow

import torch
import torch.nn as nn
import torch.optim as optim

import math
import numpy as np

from preprocess import model_wiki, prepare_batches, triplets, tokenizer, random_passage_generator, pad_sequences_left

from model import TowerD, TowerQ, cosine_distance, triplet_loss_function


import wandb






"""
prepare batched padded triplets in list format.
"""
batch_size = 32
batches_test = prepare_batches(triplets, batch_size)
# batches_train = prepare_batches(triplets_train, batch_size)
# batches_valid = prepare_batches(triplets_valid, batch_size)

print(f"Total batches: {len(batches_test)}")
for idx, (queries, docpos, docminus) in enumerate(batches_test[:10]):
    print(f"Batch {idx}:")
    print(f"  Queries shape: {queries.shape}")
    print(f"  Positive Documents shape: {docpos.shape}")
    print(f"  Negative Documents shape: {docminus.shape}")

for idx, (queries, docpos, docminus) in enumerate(batches_test[:10]):
    assert queries.size(1) > 0, f"Batch {idx} has empty query sequences."
    assert docpos.size(1) > 0, f"Batch {idx} has empty positive document sequences."
    assert docminus.size(1) > 0, f"Batch {idx} has empty negative document sequences."


# instantiate the initial model

embedding_dim = 50
hidden_dim = 100
tower_q = TowerQ(embedding_dim, hidden_dim)
tower_d = TowerD(embedding_dim, hidden_dim)


# Initialize a new run
wandb.init(project="two_tower_rnn", config={
    "embedding_dim": embedding_dim,
    "hidden_dim": hidden_dim,
    "batch_size": batch_size,
    "epochs": 5,
    "learning_rate": 0.005,
    "margin": 0.5,  # For triplet loss margin
    "log_interval": 10
})

config = wandb.config

input_size_q = config.embedding_dim  # Size of the query embeddings
input_size_d = config.embedding_dim  # Size of the document embeddings
output_size = config.hidden_dim    # Desired output size for each tower

optimizer = torch.optim.Adam(list(tower_q.parameters()) + list(tower_d.parameters()), lr=config.learning_rate)

batch_size = config.batch_size
epochs = config.epochs
batches = batches_test
total_batches = len(batches)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for epoch in range(config.epochs):
    tower_q.train()
    tower_d.train()
    running_loss = 0.0

    for batch_idx, (queries, docpos, docminus) in enumerate(batches_test):
        queries, docpos, docminus = queries.to(device), docpos.to(device), docminus.to(device)
        print(queries.shape, docpos.shape, docminus.shape)
        # Pass the batched queries, positive docs, and negative docs to the models
        query_hidden_state = tower_q(queries)
        pos_doc_hidden_state = tower_d(docpos)
        neg_doc_hidden_state = tower_d(docminus)

        # Compute the triplet loss
        loss = triplet_loss_function(query_hidden_state, pos_doc_hidden_state, neg_doc_hidden_state, margin=config.margin)

        # Backpropagation and optimization
        optimizer.zero_grad()  # Clear gradients
        loss.backward()        # Backpropagate the loss
        optimizer.step()       # Update parameters

        running_loss += loss.item()

        if (batch_idx + 1) % config.log_interval == 0:  # Add log_interval to config
          wandb.log({"batch_loss": loss.item(), "epoch": epoch, "batch_idx": batch_idx})
          print(f"Epoch [{epoch + 1}/{config.epochs}], Batch [{batch_idx + 1}/{total_batches}], Loss: {100*loss.item()}")

    # Log the average loss for the epoch
    wandb.log({"epoch": epoch, "loss": running_loss / len(batches)})
    print(f"Epoch {epoch+1}/{config.epochs}, Loss: {running_loss/len(batches)}")

# Save the model state dictionaries to Wandb
torch.save(tower_q.state_dict(), 'model_tower_query_final.pth')
torch.save(tower_d.state_dict(), 'model_tower_document_final.pth')
wandb.save('model_tower_query_final.pth')
wandb.save('model_tower_document_final.pth')
wandb.finish()


