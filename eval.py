from model import CPTG
import pandas as pd
import os
import dill

import random
import math
import numpy as np

import torch
from torchtext.legacy.data import LabelField, Field, Dataset, TabularDataset, BucketIterator, Pipeline
from torchtext.vocab import Vocab
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import embedding_loader, build_iterators , remove_redundant_tokens
from model import initialize_model

PATH = os.getcwd()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = PATH + "/model.pt"
BATCH_SIZE = 32


if __name__ == "__main__":

  dataset_name = "amazon"

  REVIEW, LABEL, train_iterator, val_iterator, test_iterator = build_iterators(dataset_name,BATCH_SIZE)
  vocab_sz = len(REVIEW.vocab)

  # evaluate and/or generate

  model = initialize_model(vocab_sz).to(device)
  model.load_state_dict(torch.load(MODEL_PATH))

  pos = []
  neg = []

  model.eval()
  for batch in val_iterator:

    X = batch.review.to(device)
    y = batch.label.to(device)

    returned_sentences = model.change_sentiment(X, y)

    print(returned_sentences.device)

    returned_sentences = returned_sentences.detach().cpu().numpy()
    original_sentences = X.detach().cpu().numpy()

    b_sz = returned_sentences.shape[0]

    for b in range(b_sz):
      print(f"Style changed sentence: {' '.join([REVIEW.vocab.itos[x] for x in returned_sentences[b]])}", end="\t")
      print(f"Original sentence: {' '.join([REVIEW.vocab.itos[x] for x in original_sentences[b]])}")
      # print(f"Ground truth: {y[b]}")
      if y[b] == 0:
        pos.append(' '.join([REVIEW.vocab.itos[x] for x in returned_sentences[b]]))
      if y[b] == 1:
        neg.append(' '.join([REVIEW.vocab.itos[x] for x in returned_sentences[b]]))

    synth = pd.DataFrame({"review": pos+neg, "label": [1]*len(pos)+[0]*len(neg)})
    synth = synth.dropna()

    synth["review"] = synth["review"].apply(remove_redundant_tokens)

    synth.to_csv("/content/gdrive/MyDrive/LAB_work/my_CPTG/data/"+dataset_name+"_Style_transferred_Synthetic_data.csv", index = False)

