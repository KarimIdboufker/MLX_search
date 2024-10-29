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
from torch.utils.data import DataLoader, TensorDataset

import math
import numpy as np

import collections
from collections import defaultdict


# import re
# import nltk
# from nltk.corpus import stopwords
# import tqdm

import gensim
import gensim.downloader



# import data from parquet file to DataFrame


model_wiki = gensim.downloader.load('glove-wiki-gigaword-50')

# create additional field as list of passages selected randomly:

def random_passage_generator(df):
    passage_list = df['passages.passage_text'].tolist()
    def generate_random_passage(passage):
        return [random.choice(random.choice(passage_list)) for _ in passage]  
    return generate_random_passage

# Extract columns ids, query, relevant examples as list, irrelevant examples as list





# custome tokenizer

def preprocess(text: str) -> list[str]:
  text = text.lower()
  text = text.replace('.',  ' <PERIOD> ')
  text = text.replace(',',  ' <COMMA> ')
  text = text.replace('"',  ' <QUOTATION_MARK> ')
  text = text.replace(';',  ' <SEMICOLON> ')
  text = text.replace('!',  ' <EXCLAMATION_MARK> ')
  text = text.replace('?',  ' <QUESTION_MARK> ')
  text = text.replace('(',  ' <LEFT_PAREN> ')
  text = text.replace(')',  ' <RIGHT_PAREN> ')
  text = text.replace('--', ' <HYPHENS> ')
  text = text.replace('?',  ' <QUESTION_MARK> ')
  text = text.replace(':',  ' <COLON> ')
  words = text.split()
  stats = collections.Counter(words)
  words = [word for word in words]
  return words

def tokenizer(text):
    """
    Tokenizes text and retrieves token IDs (word embeddings) from the Word2Vec model.
    """
    tokens = preprocess(text)  # Simple tokenization (split by spaces)
    token_ids = [model_wiki.key_to_index[token] for token in tokens if token in model_wiki]
    return token_ids


def pad_sequences_left(sequences, max_len):
    return [[0] * (max_len - len(seq)) + seq for seq in sequences]

def prepare_batches(df, batch_size):
    """
    Splits the tokenized triplets into batches and pads queries, positive docs, and negative docs.
    """
    batches = []

    for i in range(0, len(df), batch_size):
        batch = df[i:i + batch_size]

        # Tokenize queries, positive docs, and negative docs
        query_tokens = batch['query'].to_list()
        docpos_tokens = batch['docpos'].to_list()
        negdocs_tokens = batch['docminus'].to_list()


        # Find the max length of sequences in the batch (for padding)
        max_query_len = max(len(q) for q in query_tokens)
        max_doc_len = max(max(len(d) for d in docpos_tokens), max(len(n) for n in negdocs_tokens))

        # Pad queries, positive docs, and negative docs
        padded_queries = pad_sequences_left(query_tokens, max_query_len)
        padded_docpos = pad_sequences_left(docpos_tokens, max_doc_len)
        padded_neg_docs = pad_sequences_left(negdocs_tokens, max_doc_len)

        # # Convert to torch tensors for further processing in the model
        queries_tensor = torch.tensor(padded_queries, dtype=torch.long)
        docpos_tensor = torch.tensor(padded_docpos, dtype=torch.long)
        negdocs_tensor = torch.tensor(padded_neg_docs, dtype=torch.long)


        batches.append((queries_tensor, docpos_tensor, negdocs_tensor))
        
    return batches

df = pd.read_parquet('Marco_dataset/test-00000-of-00001.parquet', engine = 'fastparquet')
# df_valid = pd.read_parquet('validation-00000-of-00001.parquet', engine = 'fastparquet')
# df_train = pd.read_parquet('train-00000-of-00001.parquet', engine = 'fastparquet')

random_passage_list = random_passage_generator(df)

df_test = df[['query_id', 'query', 'query_type','passages.passage_text']]
df_test['passage_irrelevant'] = df_test['passages.passage_text'].apply(random_passage_list)

# df_valid = df_valid[['query_id', 'query', 'query_type','passages.passage_text']]
# df_valid['passage_irrelevant'] = df_valid['passages.passage_text'].apply(random_passage_list)

# df_train = df_train[['query_id', 'query', 'query_type','passages.passage_text']]
# df_train['passage_irrelevant'] = df_train['passages.passage_text'].apply(random_passage_list)

# build a dataframe of tokenized words
df_test_tokenized = pd.DataFrame()
df_test_tokenized['query'] = df_test['query'].apply(tokenizer)
df_test_tokenized['docpos'] = df_test['passages.passage_text'].apply(lambda passages: [tokenizer(passage) for passage in passages])
df_test_tokenized['docminus'] = df_test['passage_irrelevant'].apply(lambda passages: [tokenizer(passage) for passage in passages])

# df_valid_tokenized = pd.DataFrame()
# df_valid_tokenized['query'] = df_valid['query'].apply(tokenizer)
# df_valid_tokenized['docpos'] = df_valid['passages.passage_text'].apply(lambda passages: [tokenizer(passage) for passage in passages])
# df_valid_tokenized['docminus'] = df_valid['passage_irrelevant'].apply(lambda passages: [tokenizer(passage) for passage in passages])

# df_train_tokenized = pd.DataFrame()
# df_train_tokenized['query'] = df_train['query'].apply(tokenizer)
# df_train_tokenized['docpos'] = df_train['passages.passage_text'].apply(lambda passages: [tokenizer(passage) for passage in passages])
# df_train_tokenized['docminus'] = df_train['passage_irrelevant'].apply(lambda passages: [tokenizer(passage) for passage in passages])

triplets_pd = df_test_tokenized.explode(['docpos', 'docminus'])
triplets_pd.reset_index(drop=True, inplace=True)
triplets = pl.from_pandas(triplets_pd)

# triplets_train = df_train_tokenized.explode(['docpos', 'docminus'])
# triplets_train.reset_index(drop=True, inplace=True)
# triplets_train = pl.from_pandas(triplets_train)

# triplets_valid = df_valid_tokenized.explode(['docpos', 'docminus'])
# triplets_valid.reset_index(drop=True, inplace=True)
# triplets_valid = pl.from_pandas(triplets_valid)

#print(triplets.shape, triplets_train.shape, triplets_valid.shape)