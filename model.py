import random
import math

import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def one_hot(l, attr_size=2):
    # make one-hot vector representation of a label
    batch_size = l.size(0)
    one_hot = l.new_zeros(batch_size, attr_size).float()
    one_hot[range(batch_size), l] = 1
    return one_hot


def initialize_model(vocab_size):
    generator = Generator(vocab_size)
    discriminator = Discriminator()
    return CPTG(generator, discriminator)


class CPTG(nn.Module):

    def __init__(self, generator, discriminator):

        super(CPTG,self).__init__()

        self.generator = generator
        self.discriminator = discriminator

    def count_params(self):

        trainable =  sum(p.numel() for p in self.generator.parameters() if p.requires_grad) + \
                    sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
        print(f"Number of trainable parameters : {trainable}")


    def change_sentiment(self, X, y):

        hy, Y = self.generator(X, y, teacher_force = False)
        return Y # [batch * seq len]

    def forward(self, X, label):
        # X = [batch * seq len], y = [batch,]

        hy, Y = self.generator(X, label, teacher_force = False)
        # [batch, seq len, 700], [batch, seq len]


        hx, output_logits = self.generator(Y, label, teacher_force=True)
        # [batch, seq len, 700], [batch, seq len, vocab]

        y_verdict = self.discriminator(hy, 1-label)
        x_verdict = self.discriminator(hx, label)
        x_verdict_l_ = self.discriminator(hx, 1 - label)

        return x_verdict, y_verdict, x_verdict_l_, output_logits


class Generator(nn.Module):

    def __init__(self, vocab_size):

        super(Generator,self).__init__()

        self.vocab_size = vocab_size

        self.emb = nn.Embedding(vocab_size, 300)
        self.label_emb = nn.Embedding(2, 200)

        self.encoder = nn.GRU(300, 500, batch_first = True, bidirectional = False)
        self.decoder = nn.GRU(300, 700, batch_first = True, bidirectional = False)

        self.out = nn.Linear(700, self.vocab_size)

        self.gamma = 0.5


    # this does not backpropagate at all
    def _hard_sampling(self, output):
        # output = [batch *  1 * vocab_size]
        prob = F.softmax(output.squeeze(dim=1),dim=-1)
        sampled = torch.multinomial(prob, num_samples=1)
        #len_ = (sampled != EOS_IDX).squeeze(1).long()
        return sampled.detach() #(batch * 1)

    def _fuse(self):
        g = torch.empty_like(self.z_x).bernoulli_(self.gamma) # (B, 500)
        z_xy = (g * self.z_x) + ((1 - g) * self.z_y)
        return z_xy # (B, 500)

    
    def forward(self, X, l, teacher_force = True):
        # X = [batch * seq len], y = [batch,]

        MAXLEN = X.size(1)
        batch_size = X.size(0)

        emb_x = self.emb(X) # [batch * seq len * 300]
        emb_l = self.label_emb(l) # [batch, 200]
        emb_l_ = self.label_emb(torch.add(1, -1*l))

        _, h_ = self.encoder(emb_x)
        #h_x = [1, batch, 500]

        h_ = h_.squeeze(dim=0)

        if teacher_force: # loss computation with teacher forcing, regenerating X
            
            self.z_y = h_.clone()

            z_xy = self._fuse()

            hidden = torch.cat((z_xy, emb_l), dim=-1).unsqueeze(dim=0) # z_y = [1 * batch * 700]
            x = X[:,:-1] # removed "<eos>" token
            x_embed = self.emb(x) # (batch * seq len - 1 * 300)
            hx, _ = self.decoder(x_embed, hidden)
            #hx = [batch * seq len - 1 * 700]

            # hx, lengths = pad_packed_sequence(packed_out, batch_first=True,
            #                                   total_length=total_length)

            output_logits = self.out(hx)
            return hx , output_logits # (batch * seq len * 700), (batch * seq len * vocab)
        
        
        else: # sample y
            
            self.z_x = h_.clone()

            hidden = torch.cat((self.z_x, emb_l_), dim=-1).unsqueeze(dim=0) # z_x = [1 * batch * 700]
            y = []
            hy = []
            input_ = l.new_full((batch_size, 1), 2) # <sos> token at first

            for t in range(MAXLEN-1):
                input_ = self.emb(input_) # (B, 1, 300)
                # output (B, 1, 700), hidden (1, B, 700)
                output, hidden = self.decoder(input_, hidden)
                input_ = self._hard_sampling(self.out(output))
                hy.append(output)
                y.append(input_)
            input_ = l.new_full((batch_size,1), 3) # feed <eos> as last input,
            # output, _ = self.gru(self.emb(input_), hidden)
            hy.append(output)
            y.append(input_) # append <eos> as last token

            hy = torch.cat(hy, dim=1)
            y = torch.cat(y, dim=1)
            # hy, y, lengths = self._tighten(hy, y)
            #lengths = y.new_full((B,), MAXLEN+1)
            return hy, y # [batch * seq len * 700] * [batch * seq len] [ tok1, tok2, ... , <eos>]
        


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator,self).__init__()
        self.encoder = nn.GRU(700, 500, batch_first=True, bidirectional=True)
        self.projector1 = nn.Linear(500*2 , 1)
        self.projector2 = nn.Linear(500*2 , 2, bias = False)
    
    def forward(self, h, l):
        # h = [batch * seq len * 700], l = [batch,]

        l = one_hot(l)

        _, encoded = self.encoder(h) # [(directions*layers) * batch * 500]

        e = torch.cat((encoded[0],encoded[1]),dim=-1) # [batch * (500*2)]

        p1 = self.projector1(e)
        p2 = self.projector2(e)

        verdict = p1 + torch.sum(l * p2, dim=-1).unsqueeze(dim=-1) # real/fake verdict
        
        return verdict # [batch, 1]





