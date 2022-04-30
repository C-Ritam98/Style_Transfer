from torchtext.vocab import FastText
import os
# embedding_glove = GloVe(name='6B', dim=100)

import torch
from torchtext.legacy.data import LabelField, Field, Dataset, TabularDataset, BucketIterator, Pipeline
from torchtext.vocab import Vocab

import pandas as pd
import dill

import random
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PATH = os.getcwd()

def embedding_loader(type = "fasttext"):

    cache = PATH + "/vector_cache"

    if not os.path.exists(cache):
        os.mkdir(cache)
        
    if type == "fasttext":
        vectors = FastText(language='en', cache=cache)
    
    if type == "glove":
        vectors = GloVe(name='6B', cache=cache)
    

    if type == "word2vec":
        vectors = Word2vec(language='en', cache=cache)

    return vectors




def build_iterators(dataset="yelp",BATCH_SIZE = 32, return_review_object = True):

	if dataset.lower() == "yelp":
		dataset = "Yelp"
	if dataset.lower() == "imdb":
		dataset = "IMDB"
	if dataset.lower() == "amazon":
		dataset = "Amazon"

	REVIEW = Field(sequential=True,
	      tokenize = lambda x: x.split(),
	      use_vocab = True, 
	      init_token = '<sos>', 
	      eos_token = '<eos>', 
	      pad_token = '<pad>',
	      # fix_length = 22,
	      batch_first = True,
	      lower = True)

	LABEL = Field(sequential=False, dtype=torch.int64, batch_first=True, use_vocab=False,\
		   preprocessing=Pipeline(lambda x: int(x)))

	fields = [('review', REVIEW),('label',LABEL)]

	train_data, val_data, test_data = TabularDataset.splits(
		                        path = PATH,
		                        train = './data/'+dataset+'/train.csv',
		                        validation = './data/'+dataset+'/val.csv',
		                        test = './data/'+dataset+'/test.csv',
		                        format = 'csv',
		                        fields = fields,
		                        skip_header = True)

	vectors = embedding_loader("fasttext")

	REVIEW.build_vocab(train_data, val_data, test_data, vectors = vectors, min_freq = 2)
	LABEL.build_vocab(train_data)

	print("Meta Data :")
	print(f"Vocab size: {len(REVIEW.vocab)}")
	print(f"<eos> index: {REVIEW.vocab.stoi['<eos>']}")
	print(f"<sos> index: {REVIEW.vocab.stoi['<sos>']}")
	print(f"<pad> index: {REVIEW.vocab.stoi['<pad>']}")
	print(f"<unk> index: {REVIEW.vocab.stoi['<unk>']}")

	# print("Training data instance:")
	# print(train_data[0].__dict__.keys())
	# print(train_data[21].__dict__.values())

	train_iterator, val_iterator, test_iterator = BucketIterator.splits(
		                                        (train_data, val_data, test_data), 
		                                        batch_size = BATCH_SIZE,
		                                        sort_key = lambda x : len(x.review),
		                                        shuffle = False,
		                                        device = device
		                                        )

	if return_review_object:
	  return REVIEW, LABEL, train_iterator, val_iterator, test_iterator
	else:
	  return train_iterator, val_iterator, test_iterator



def remove_redundant_tokens(x):
  return ' '.join([tok for tok in x.split() if tok not in ["<unk>","<sos>","<eos>","<pad>"]])



def Create_data(dataset = "Yelp"):

  print(f"Current directory: {os.getcwd()}")

  PATH = os.getcwd() #"/media/rahul/DATA-2/my_CPTG"

  if dataset.lower() == "yelp":
    dataset = "Yelp"
  if dataset.lower() == "imdb":
    dataset = "IMDB"
  if dataset.lower() == "amazon":
    dataset = "Amazon"        

  if os.path.exists(PATH+'/data/'+dataset+'/train.csv') and os.path.exists(PATH+'/data/'+dataset+'/val.csv') and os.path.exists(PATH+'/data/'+dataset+'/test.csv'):
    # use pre-built datasets
    return

  # PATH = "/home/cds-2/Desktop/ri/CPTG/data/"

  with open(PATH+"/data/sentiment.train.1",'r') as f:
    pos_data = f.readlines()[:10000]
    pos_data = [x.split('\n')[0] for x in pos_data]


  with open(PATH+"/data/sentiment.train.0",'r') as f:
    neg_data = f.readlines()[:10000]
    neg_data = [x.split('\n')[0] for x in neg_data]

  print(f"Pos size = {len(pos_data)}, Neg size = {len(neg_data)}")

  df_train = pd.DataFrame({"review":pos_data[:8000]+neg_data[:8000], "label":[1]*8000 + [0]*8000})
  df_val = pd.DataFrame({"review":pos_data[8000:9000]+neg_data[8000:9000], "label":[1]*1000 + [0]*1000})
  df_test = pd.DataFrame({"review":pos_data[9000:]+neg_data[9000:], "label":[1]*1000 + [0]*1000})

  df_train = df_train.sample(frac=1)
  # df_val = df_val.sample(frac=1)
  # df_test = df_test.sample(frac=1)

  PATH = os.getcwd()

  df_train.to_csv("train.csv",index=False)
  df_val.to_csv("val.csv",index=False)
  df_test.to_csv("test.csv",index=False)

