import pandas as pd
import os
import dill

import random
import math
import pickle
import matplotlib.pyplot as plt

import numpy as np
import torch
from torchtext.legacy.data import LabelField, Field, Dataset, TabularDataset, BucketIterator, Pipeline
from torchtext.vocab import Vocab
import torch.nn as nn
import torch.optim as optim

from utils import embedding_loader, build_iterators, Create_data
from model import initialize_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    PATH = os.getcwd()
    
    dataset_name = "amazon"

    if PATH.split('/')[-1] != "my_CPTG":
      print(f"Currently in :{PATH}")
      print("Please move to the directiory --> ./my_CPTG")
      exit()

    MODEL_PATH = PATH + "/model.pt"

    print("[INFO]...Creating data...")

    Create_data(dataset_name)

    print("[INFO]...Data created!")
    
    print(f"CUDA availability: {torch.cuda.is_available()}")
    
    REVIEW, LABEL, train_iterator, val_iterator, test_iterator = build_iterators(dataset_name)

    model = initialize_model(len(REVIEW.vocab)).to(device)

    discriminator_criterion = nn.BCEWithLogitsLoss()
    generator_criterion = nn.CrossEntropyLoss(ignore_index=1,reduction='mean')

    generator_optimizer = optim.Adam(model.generator.parameters(), lr = 3e-4)
    discriminator_optimizer = optim.Adam(model.discriminator.parameters(), lr = 3e-4)

    model.generator.emb.weight.data.copy_(REVIEW.vocab.vectors)
    model.generator.emb.weight.requires_grad = False

    model.count_params()

    print("[INFO]...Training started...")

    # Training

    N_EPOCHS = 50
    var_gen_loss_list = []
    var_disc_loss_list = []
    
    for epoch in range(N_EPOCHS):

      model.train()

      var_gen_loss , var_disc_loss = 0, 0

      for batch in train_iterator:

        X = batch.review.to(device)
        y = batch.label.to(device)

        batch_size = X.size(0)
        seq_len = X.size(1)

        x_verdict, y_verdict, x_verdict_l_, output_logits = model(X,y)
        
        ones = y.new_ones(x_verdict.size()).detach()
        zeros = y.new_ones(x_verdict.size()).detach()

        disc_loss = 0.5 * (2*discriminator_criterion(x_verdict, ones.float()) \
                     + discriminator_criterion(y_verdict, zeros.float()) \
                     + discriminator_criterion(x_verdict_l_, zeros.float()))

        discriminator_optimizer.zero_grad()
        disc_loss.backward()
        discriminator_optimizer.step()
        discriminator_optimizer.zero_grad()
        generator_optimizer.zero_grad()


        x_verdict, y_verdict, x_verdict_l_, output_logits = model(X,y)

        X_truncated = X[:,1:].clone() # removing the <sos> tokens for comparing with the new data

        gen_loss = generator_criterion(output_logits.contiguous().view(batch_size*(seq_len-1),-1), X_truncated.contiguous().view(-1))

        disc_loss = discriminator_criterion(y_verdict, ones.float()) \
                    + discriminator_criterion(x_verdict_l_, ones.float())

        generator_optimizer.zero_grad()
        tot_gen_loss = gen_loss + 0.5 * disc_loss
        tot_gen_loss.backward()
        generator_optimizer.step()
        generator_optimizer.zero_grad()

        var_gen_loss += tot_gen_loss.item()
        var_disc_loss += disc_loss.item()
        

      print(f"Epoch : {epoch + 1} | Gen Loss: {var_gen_loss} | Disc Loss: {var_disc_loss}")
      var_gen_loss_list.append(var_gen_loss)
      var_disc_loss_list.append(var_disc_loss)

      try:
        torch.save(model.state_dict(), MODEL_PATH)
        print("[INFO]...Model saved...")
      except:
        print("[ALERT]...Unable to save model!")    
    
    plt.plot(var_gen_loss_list,label = "Generator Loss")
    plt.plot(var_disc_loss_list, label = "Disc Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig(PATH+"/Loss.png")

    print("[INFO]...Training complete!")
    print("Run ./eval.py to get a glimpse of the trained model.")









