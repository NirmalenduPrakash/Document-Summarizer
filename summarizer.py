#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 17:18:42 2020

@author: nirmalenduprakash
Document summarization using pointer generator network
Dataset downloaded from https://github.com/harvardnlp/sent-summary
pretrined embeddings used-word2vec.6B.100d(fixed during training)
vocab created from training set(words with frequency>20)
"""

import torch
import torch.nn as nn
import math
from torch import optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from typing import NamedTuple, List, Callable, Dict, Tuple, Optional
from collections import Counter
from random import shuffle
import numpy as np
import os
import re
import subprocess
import gzip
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from copy import deepcopy
import pickle

class Params:
  hidden_size: int = 150  # of the encoder; default decoder size is doubled if encoder is bidi
  dec_hidden_size: Optional[int] = 200  # if set, a matrix will transform enc state into dec state
  embed_size: int = 100
  
  # Data
  embed_file: Optional[str] = '/home/svu/e0401988/NLP/summarization/word2vec.6B.100d.txt'  # use pre-trained embeddings
  data_path: str = '/home/svu/e0401988/NLP/summarization/cnndm.gz'
  val_data_path: Optional[str] = '/home/svu/e0401988/NLP/summarization/cnndm.val.gz'
  max_src_len: int = 400  # exclusive of special tokens such as EOS
  max_tgt_len: int = 100  # exclusive of special tokens such as EOS
  eps=1e-31
  batch_size=16
  encoder_weights_path='/home/svu/e0401988/NLP/summarization/encoder_sum.pt'
  decoder_weights_path='/home/svu/e0401988/NLP/summarization/decoder_sum.pt'
  encoder_decoder_adapter_weights_path='/home/svu/e0401988/NLP/summarization/adapter_sum.pt'
  losses_path='/home/svu/e0401988/NLP/summarization/val_losses.pkl'

  # Testing
  test_data_path: str = '/content/drive/My Drive/cnndm.test.gz'
  
def simple_tokenizer(text, lower=False, newline=None):
  if lower:
    text = text.lower()
  if newline is not None:  # replace newline by a token
    text = text.replace('\n', ' ' + newline + ' ')
  return text.split()

def count_words(count_dict, text):
    '''Count the number of occurrences of each word in a set of text'''
    for sentence in text:
        for word in sentence:
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1

class Vocab(object):
  PAD = 0
  SOS = 1
  EOS = 2
  UNK = 3

  def __init__(self):
    self.word2index = {}
    self.word2count = Counter()
    self.reserved = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    self.index2word = self.reserved[:]
    self.embeddings = None

  def add_words(self, words):
    for word in words:
      if word not in self.word2index:
        self.word2index[word] = len(self.index2word)
        self.index2word.append(word)
    self.word2count.update(words)
  
  def load_embeddings(self, file_path: str, dtype=np.float32) -> int:
    num_embeddings = 0
    vocab_size = len(self)
    with open(file_path, 'rb') as f:
      for line in f:
        line = line.split()
        word = line[0].decode('utf-8')
        # self.add_words([word])
        idx = self.word2index.get(word)
        if idx is not None:
          vec = np.array(line[1:], dtype=dtype)
          if self.embeddings is None:
            n_dims = len(vec)
            self.embeddings = np.random.normal(np.zeros((vocab_size, n_dims))).astype(dtype)
            self.embeddings[self.PAD] = np.zeros(n_dims)
          self.embeddings[idx] = vec          
          num_embeddings += 1
    return num_embeddings

  def __getitem__(self, item):
    if type(item) is int:
      return self.index2word[item]
    return self.word2index.get(item, self.UNK)

  def __len__(self):
    return len(self.index2word)

class Dataset(object):

  def __init__(self, filename: str, tokenize: Callable=simple_tokenizer, max_src_len: int=None,
               max_tgt_len: int=None, truncate_src: bool=False, truncate_tgt: bool=False):
    print("Reading dataset %s..." % filename, end=' ', flush=True)
    self.filename = filename
    self.pairs = []
    self.src_len = 0
    self.tgt_len = 0
    if filename.endswith('.gz'):
      open = gzip.open
    with open(filename, 'rt', encoding='utf-8') as f:
      for i, line in enumerate(f):
        # if(i<1000):
        pair = line.strip().split('\t')
        if len(pair) != 2:
          print("Line %d of %s is malformed." % (i, filename))
          continue
        src = tokenize(pair[0])
        if max_src_len and len(src) > max_src_len:
          if truncate_src:
            src = src[:max_src_len]
          else:
            continue
        tgt = tokenize(pair[1])
        if max_tgt_len and len(tgt) > max_tgt_len:
          if truncate_tgt:
            tgt = tgt[:max_tgt_len]
          else:
            continue
        src_len = len(src) + 1  # EOS
        tgt_len = len(tgt) + 1  # EOS
        self.src_len = max(self.src_len, src_len)
        self.tgt_len = max(self.tgt_len, tgt_len)
        self.pairs.append((src, tgt, src_len, tgt_len))
    print("%d pairs." % len(self.pairs))

  def build_vocab(self, embed_file: str=None) -> Vocab:
    # word frequency
    word_counts={}
    count_words(word_counts,[src+tgr for src,tgr,len_src,len_tgr in self.pairs])
    vocab=Vocab()
    for word,count in word_counts.items():
        if(count>20):
            vocab.add_words([word])  
    count = vocab.load_embeddings(embed_file)
    print("%d pre-trained embeddings loaded." % count)

    return vocab  

def get_coverage_vector(enc_attn_weights):
    """Combine the past attention weights into one vector"""
    # if self.cover_func == 'max':
      # coverage_vector, _ = torch.max(torch.cat(enc_attn_weights), dim=0)
    # elif self.cover_func == 'sum':
    coverage_vector = torch.sum(torch.cat(enc_attn_weights), dim=0)
    # else:
      # raise ValueError('Unrecognized cover_func: ' + self.cover_func)
    return coverage_vector  

   
class CNNDataset(nn.Module):
    def __init__(self, src_sents, trg_sents,vocab):     
      self.src_sents = src_sents
      self.trg_sents = trg_sents
      self.vocab=vocab
      # Keep track of how many data points.
      self._len = len(src_sents)

      # max_src_len = max(len(sent) for sent in src_sents)
      # max_trg_len = max(len(sent) for sent in trg_sents)
      # self.max_len = max(max_src_len, max_trg_len)
        
    # def pad_sequence(self, vectorized_sent, max_len):
    #     pad_dim = (0, max_len - len(vectorized_sent))
    #     return F.pad(vectorized_sent, pad_dim, 'constant')
        
    def __getitem__(self, index):
        # vectorized_src = self.vectorize(self.vocab, self.src_sents[index])
        # vectorized_trg = self.vectorize(self.vocab, self.trg_sents[index])
        
        # return {'x':self.pad_sequence(vectorized_src, 400), 
        #         'y':self.pad_sequence(vectorized_trg, 100), 
        #         'x_len':len(vectorized_src), 
        #         'y_len':len(vectorized_trg)}
        return {'x':self.src_sents[index], 
                'y':self.trg_sents[index], 
                'x_len':len(self.src_sents[index]), 
                'y_len':len(self.trg_sents[index])}
    
    def __len__(self):
        return self._len
         
class EncoderRNN(nn.Module):
  def __init__(self, embed_size, hidden_size, bidi=True, rnn_drop: float=0):
    super(EncoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.num_directions = 2 if bidi else 1
    self.gru = nn.GRU(embed_size, hidden_size, bidirectional=bidi, dropout=rnn_drop)
    # self.hidden=torch.zeros(self.num_directions, batch_size, self.hidden_size, device=DEVICE)

  def forward(self, embedded,hidden,input_lengths=None):
    if input_lengths is not None:
      embedded = pack_padded_sequence(embedded, input_lengths,batch_first=True)
    output, hidden = self.gru(embedded,hidden)

    if input_lengths is not None:
      output, _ = pad_packed_sequence(output)

    if self.num_directions > 1:
      # hidden: (num directions, batch, hidden) => (1, batch, hidden * 2)
      batch_size = hidden.size(1)
      hidden = hidden.transpose(0, 1).contiguous().view(1, batch_size,
                                                        self.hidden_size * self.num_directions)
    return output, hidden
  def init_hidden(self, batch_size):
    return torch.zeros(self.num_directions, batch_size, self.hidden_size, device=DEVICE) 

class DecoderRNN(nn.Module):

  def __init__(self, vocab_size, embed_size, hidden_size, enc_attn=True, dec_attn=True,
               enc_attn_cover=True, pointer=True,
               in_drop: float=0, rnn_drop: float=0, out_drop: float=0, enc_hidden_size=None):
    super(DecoderRNN, self).__init__()
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.combined_size = hidden_size

    self.in_drop = nn.Dropout(in_drop) if in_drop > 0 else None
    self.gru = nn.GRU(embed_size, hidden_size, dropout=rnn_drop)

    # if enc_attn:
    if not enc_hidden_size: enc_hidden_size = self.hidden_size
    self.enc_bilinear = nn.Bilinear(hidden_size, enc_hidden_size, 1)
    self.combined_size += enc_hidden_size
    if enc_attn_cover:
      self.cover_weight = nn.Parameter(torch.rand(1))

    # if dec_attn:
    self.dec_bilinear = nn.Bilinear(self.hidden_size, self.hidden_size, 1)
    self.combined_size += self.hidden_size

    self.out_drop = nn.Dropout(out_drop) if out_drop > 0 else None
    # if pointer:
    self.ptr = nn.Linear(self.combined_size, 1)

    # if tied_embedding is not None and embed_size != self.combined_size:
    #   self.out_embed_size = embed_size

    # if self.out_embed_size: 
    #   self.pre_out = nn.Linear(self.combined_size, self.out_embed_size)
    #   size_before_output = self.out_embed_size
    # else: 
    #   size_before_output = self.combined_size

    self.out = nn.Linear(self.combined_size, vocab_size)
    # if tied_embedding is not None:
    #   self.out.weight = tied_embedding.weight

  def forward(self, embedded, hidden, encoder_hidden=None, decoder_states=None, coverage_vector=None, *,
              encoder_word_idx=None, ext_vocab_size: int=None, log_prob: bool=True):
    batch_size = embedded.size(0)
    combined = torch.zeros(batch_size, self.combined_size, device=DEVICE)

    if self.in_drop: embedded = self.in_drop(embedded)

    output, hidden = self.gru(embedded.unsqueeze(0), hidden)  # unsqueeze and squeeze are necessary
    combined[:, :self.hidden_size] = output.squeeze(0)        # as RNN expects a 3D tensor (step=1)
    offset = self.hidden_size
    enc_attn, prob_ptr = None, None  # for visualization

    # if self.enc_attn or self.pointer:
    # energy and attention: (num encoder states, batch size, 1)
    num_enc_steps = encoder_hidden.size(0)
    enc_total_size = encoder_hidden.size(2)
    # print(encoder_hidden.shape)
    enc_attn = self.enc_bilinear(hidden.expand(num_enc_steps, batch_size, -1).contiguous(),encoder_hidden)
    # print(hidden.shape,hidden.expand(num_enc_steps, batch_size, -1).shape,encoder_hidden.shape,enc_attn.shape)

    if coverage_vector is not None:
        enc_attn += self.cover_weight * torch.log(coverage_vector.transpose(0, 1).unsqueeze(2) + eps)
    # transpose => (batch size, num encoder states, 1)
    enc_attn = F.softmax(enc_attn, dim=0).transpose(0, 1)
    # if self.enc_attn:
      # context: (batch size, encoder hidden size, 1)
    enc_context = torch.bmm(encoder_hidden.permute(1, 2, 0), enc_attn)
    # print(enc_context.shape,enc_context.squeeze(2).shape)
    combined[:, offset:offset+enc_total_size] = enc_context.squeeze(2)
    offset += enc_total_size
    enc_attn = enc_attn.squeeze(2)

    # if self.dec_attn:
    if decoder_states is not None and len(decoder_states) > 0:
      dec_attn = self.dec_bilinear(hidden.expand_as(decoder_states).contiguous(),
                                      decoder_states)
      dec_attn = F.softmax(dec_attn, dim=0).transpose(0, 1)
      dec_context = torch.bmm(decoder_states.permute(1, 2, 0), dec_attn)
      combined[:, offset:offset + self.hidden_size] = dec_context.squeeze(2)
      offset += self.hidden_size
    # print(combined.shape,hidden.shape,enc_context.shape)
    # if self.out_drop: 
    # combined = self.out_drop(combined)

    # generator
    # if self.out_embed_size:
    #   out_embed = self.pre_out(combined)
    # else:
    out_embed = combined
    logits = self.out(out_embed)  # (batch size, vocab size)

    # pointer
    # if self.pointer:
    # output = torch.zeros(batch_size, self.vocab_size, device=DEVICE)
    # distribute probabilities between generator and pointer
    prob_ptr = F.sigmoid(self.ptr(combined))  # (batch size, 1)
    prob_gen = 1 - prob_ptr
    # add generator probabilities to output
    gen_output = F.softmax(logits, dim=1)  # can't use log_softmax due to adding probabilities
    output = prob_gen * gen_output
    
    pad_dim = (0, ext_vocab_size - output.size(1))
    # print(ext_vocab_size,output.size(1),pad_dim)
    output=F.pad(output, pad_dim, 'constant')

    # add pointer probabilities to output
    ptr_output = enc_attn
    # print(output.shape,encoder_word_idx.shape,prob_ptr.shape,ptr_output.shape)
    output.scatter_add_(1, encoder_word_idx, prob_ptr * ptr_output)
    # if log_prob: 
    output = torch.log(output + eps)
    # else:
    # if log_prob: output = F.log_softmax(logits, dim=1)
    # else: output = F.softmax(logits, dim=1)

    return output, hidden, enc_attn, prob_ptr 

def sort_batch_by_len(data_dict,vocab):
    data=[]
    res={'x':[],'y':[],'x_len':[],'y_len':[]}
    for i in range(data_dict['x_len']):
      data.append(preprocess(data_dict['x'][i],data_dict['y'][i],vocab))
    for i in range(len(data)):
      res['x'].append(data[i]['x'])
      res['y'].append(data[i]['y'])
      res['x_len'].append(len(data[i]['x']))
      res['y_len'].append(len(data[i]['y']))  
    
    # Sort indices of data in batch by lengths.
    sorted_indices = np.array(res['x_len']).argsort()[::-1].tolist()
    
    data_batch = {name:[_tensor[i] for i in sorted_indices]
                  for name, _tensor in res.items()}
    return data_batch

def pad_sequence(vectorized_sent, max_len):
      pad_dim = (0, max_len - len(vectorized_sent))
      return F.pad(vectorized_sent, pad_dim, 'constant').tolist()

def vectorize(vocab, tokens):
    return torch.tensor([vocab[token] for token in tokens])

def unvectorize(vocab, indices):
    return [vocab[i] for i in indices]

def preprocess(x,y,vocab):
  vectorized_src = vectorize(vocab, x)
  vectorized_trg = vectorize(vocab, y)  
  return {'x':pad_sequence(vectorized_src, 400), 
          'y':pad_sequence(vectorized_trg, 100), 
          'x_len':len(vectorized_src), 
          'y_len':len(vectorized_trg)}

def train(dataset,val_dataset,v,embedding_weights):
    p = Params()
    eps = p.eps
    batch_size =p.batch_size
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_dec_adapter = nn.Linear(p.hidden_size * 2, p.dec_hidden_size).to(DEVICE)
    embedding = nn.Embedding(len(v), p.embed_size, padding_idx=v.PAD,_weight=embedding_weights).to(DEVICE)
    embedding.weight.requires_grad=False
    encoder = EncoderRNN(p.embed_size, p.hidden_size, p.enc_bidi,rnn_drop=p.enc_rnn_dropout).to(DEVICE)
    decoder = DecoderRNN(len(v), p.embed_size, p.dec_hidden_size,
                                  enc_attn=p.enc_attn, dec_attn=p.dec_attn,
                                  pointer=p.pointer,
                                  in_drop=p.dec_in_dropout, rnn_drop=p.dec_rnn_dropout,
                                  out_drop=p.dec_out_dropout, enc_hidden_size=p.hidden_size * 2).to(DEVICE)
    if(os.path.exists(p.encoder_weights_path)):
        encoder.load_state_dict(torch.load(p.encoder_weights_path,map_location=torch.device(DEVICE)))
        decoder.load_state_dict(torch.load(p.decoder_weights_path,map_location=torch.device(DEVICE)))
        enc_dec_adapter.load_state_dict(torch.load(p.encoder_decoder_adapter_weights_path,map_location=torch.device(DEVICE)))
        # embedding.load_state_dict(torch.load('/home/svu/e0401988/NLP/summarization/embedding_sum.pt',map_location=torch.device(DEVICE)))
    # forward
    training_pairs=dataset.pairs
    cnn_data=CNNDataset([pair[0] for pair in training_pairs],[pair[1] for pair in training_pairs],v)
    validation_pairs=val_dataset.pairs
    val_data=CNNDataset([pair[0] for pair in validation_pairs],[pair[1] for pair in validation_pairs],v)
    # print(cnn_data[:3]['x_len'])
    learning_rate=0.001
    num_epochs = 10
    criterion = nn.NLLLoss(ignore_index=v.PAD)
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    adapter_optimizer=optim.SGD([{'params':enc_dec_adapter.parameters()}], lr=learning_rate)
    # embedding_optimizer=optim.SGD([{'params':embedding.parameters()}], lr=learning_rate)
    dataloader = DataLoader(dataset=cnn_data, 
                            batch_size=batch_size, 
                            shuffle=True)
    
    losses=[]
    val_losses=[]
    if(os.path.exists(p.losses_path)):
      with open(p.losses_path,'rb') as f:
        val_losses=pickle.load(f)
    # torch.cuda.empty_cache()
    for _e in range(num_epochs):
        i=0
        while i<len(cnn_data):
            try:
              data_dict=cnn_data[i:i+batch_size]
              # val_dict=val_data[i:i+batch_size]
            except:
              data_dict=cnn_data[i:len(cnn_data)]  
              # val_dict=val_data[i:len(cnn_data)]
            vocab=deepcopy(v)
            
            # print(data_dict['x_len'])  
            data_batch = sort_batch_by_len(data_dict,vocab)
    
            for word in data_dict['x']:
              vocab.add_words(word)
            data_batch_extra=sort_batch_by_len(data_dict,vocab)
            x_extra=torch.tensor(data_batch_extra['x']).to(DEVICE)
    
            x, x_len = torch.tensor(data_batch['x']).to(DEVICE), torch.tensor(data_batch['x_len']).to(DEVICE)
            y, y_len = torch.tensor(data_batch['y']).to(DEVICE), torch.tensor(data_batch['y_len']).to(DEVICE)
    
            # print(x.shape,y.shape,x_len,y_len)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            adapter_optimizer.zero_grad()
            # embedding_optimizer.zero_grad()
    
            # encoder_hidden = encoder.init_hidden(x.size(0))
            encoder_embedded = embedding(x)
            encoder_hidden=encoder.init_hidden(x.size(0))
            encoder_outputs, encoder_hidden =encoder(encoder_embedded,encoder_hidden,x_len)
            decoder_input = torch.tensor([v.SOS] * x.size(0), device=DEVICE)
            decoder_hidden = enc_dec_adapter(encoder_hidden)
            
            decoder_states = []
            enc_attn_weights = []
            loss=0
            for di in range(y.size(1)):
                decoder_embedded = embedding(decoder_input)
                if enc_attn_weights:
                    coverage_vector = get_coverage_vector(enc_attn_weights)
                else:
                    coverage_vector = None
                decoder_output, decoder_hidden, dec_enc_attn, dec_prob_ptr = decoder(decoder_embedded, decoder_hidden, encoder_outputs,
                            torch.cat(decoder_states) if decoder_states else None, coverage_vector,
                            encoder_word_idx=x_extra,log_prob=True,ext_vocab_size=len(vocab))  
                # decoder_output.to(DEVICE)
                # decoder_hidden.to(DEVICE)
                # dec_enc_attn.to(DEVICE)
                # dec_prob_ptr.to(DEVICE)
    
                decoder_states.append(decoder_hidden)      
                prob_distribution = torch.exp(decoder_output)# if log_prob else decoder_output
                # print(prob_distribution.shape)
                # top_idx = torch.multinomial(prob_distribution, 1)
                # top_idx = top_idx.squeeze(1).detach()  # detach from history as input
                _, top_idx = decoder_output.data.topk(1)
                # print(top_idx)      
                # compute loss
                # if target_tensor is None:
                #   gold_standard = top_idx  # for sampling
                # else:
                
                gold_standard = y[:,di]
                # print(y.shape)
                # if not log_prob:
                #   decoder_output = torch.log(decoder_output + eps) 
                nll_loss= criterion(decoder_output, gold_standard)    
                loss+=nll_loss
                decoder_input = y[:,di]
                if (coverage_vector is not None and criterion): #and cover_loss > 0:
                    coverage_loss = torch.sum(torch.min(coverage_vector, dec_enc_attn)) / batch_size #* cover_loss            
                    loss+=coverage_loss
                enc_attn_weights.append(dec_enc_attn.unsqueeze(0))                    
            loss.backward()
            clip_grad_norm_(encoder.parameters(), 1)
            clip_grad_norm_(decoder.parameters(), 1)
            clip_grad_norm_(enc_dec_adapter.parameters(), 1)
            clip_grad_norm_(embedding.parameters(), 1)
            encoder_optimizer.step()
            decoder_optimizer.step()
            adapter_optimizer.step() 
            # embedding_optimizer.step()
            i+=batch_size
        loss=loss.data.item()/x.size(0)
        # calculating validation loss
        
        val_loss=0
        i=0
        while(i<len(val_data)):
            # torch.cuda.empty_cache()
            try:
              val_dict=val_data[i:i+batch_size]
            except: 
              val_dict=val_data[i:len(val_data)]
            
            vocab=deepcopy(v)
            data_batch = sort_batch_by_len(val_dict,vocab)
    
            for word in val_dict['x']:
              vocab.add_words(word)
            data_batch_extra=sort_batch_by_len(data_dict,vocab)
            x_extra=torch.tensor(data_batch_extra['x']).to(DEVICE)
    
            x, x_len = torch.tensor(data_batch['x']).to(DEVICE), torch.tensor(data_batch['x_len']).to(DEVICE)
            y, y_len = torch.tensor(data_batch['y']).to(DEVICE), torch.tensor(data_batch['y_len']).to(DEVICE)
    
            # print(x.shape,y.shape,x_len,y_len)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            adapter_optimizer.zero_grad()
            # embedding_optimizer.zero_grad()
    
            # encoder_hidden = encoder.init_hidden(x.size(0))
            encoder_embedded = embedding(x)
            encoder_hidden=encoder.init_hidden(x.size(0))
            encoder_outputs, encoder_hidden =encoder(encoder_embedded,encoder_hidden,x_len)
            decoder_input = torch.tensor([v.SOS] * x.size(0), device=DEVICE)
            decoder_hidden = enc_dec_adapter(encoder_hidden)
            
            decoder_states = []
            enc_attn_weights = []
            
            for di in range(y.size(1)):
                decoder_embedded = embedding(decoder_input)
                if enc_attn_weights:
                    coverage_vector = get_coverage_vector(enc_attn_weights)
                else:
                    coverage_vector = None
                
                decoder_output, decoder_hidden, dec_enc_attn, dec_prob_ptr = decoder(decoder_embedded, decoder_hidden, encoder_outputs,
                            torch.cat(decoder_states) if decoder_states else None, coverage_vector,
                            encoder_word_idx=x_extra,log_prob=True,ext_vocab_size=len(vocab))  
                # decoder_output.to(DEVICE)
                # decoder_hidden.to(DEVICE)
                # dec_enc_attn.to(DEVICE)
                # dec_prob_ptr.to(DEVICE)
    
                decoder_states.append(decoder_hidden)      
                prob_distribution = torch.exp(decoder_output)# if log_prob else decoder_output
                # print(prob_distribution.shape)
                # top_idx = torch.multinomial(prob_distribution, 1)
                # top_idx = top_idx.squeeze(1).detach()  # detach from history as input
                _, top_idx = decoder_output.data.topk(1)
                # print(top_idx)      
                # compute loss
                # if target_tensor is None:
                #   gold_standard = top_idx  # for sampling
                # else:
                
                gold_standard = y[:,di]
                # print(y.shape)
                # if not log_prob:
                #   decoder_output = torch.log(decoder_output + eps) 
                nll_loss= criterion(decoder_output, gold_standard)    
                val_loss+=nll_loss.data.item()
                # print(y[:,di].shape,top_idx.shape)
                decoder_input = top_idx.view(-1)#y[:,di]
                if (coverage_vector is not None and criterion): #and cover_loss > 0:
                    coverage_loss = torch.sum(torch.min(coverage_vector, dec_enc_attn)) / batch_size #* cover_loss            
                    val_loss+=coverage_loss.data.item()
                enc_attn_weights.append(dec_enc_attn.unsqueeze(0))         
            i+=batch_size    
        avg_val_loss=val_loss/len(val_data)        
        print('training loss:{}'.format(loss),'validation loss:{}'.format(avg_val_loss))
        if(len(val_losses)>0 and avg_val_loss<min(val_losses)):
            torch.save(encoder.state_dict(), p.encoder_weights_path)
            torch.save(decoder.state_dict(), p.decoder_weights_path)
            torch.save(enc_dec_adapter.state_dict(), p.encoder_decoder_weights_path)
            # torch.save(embedding.state_dict(), '/home/svu/e0401988/NLP/summarization/embedding_sum.pt')
        val_losses.append(avg_val_loss) 
    
    with open(p.losses_path,'wb') as f:
        pickle.dump(val_losses,f) 
        
def predict(sent,v,batch_size=1):
    p=Params()
    eps=p.eps
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    embedding_weights = torch.from_numpy(v.embeddings)
    enc_dec_adapter = nn.Linear(p.hidden_size * 2, p.dec_hidden_size).to(DEVICE)
    embedding = nn.Embedding(len(v), p.embed_size, padding_idx=v.PAD,_weight=embedding_weights).to(DEVICE)
    encoder = EncoderRNN(p.embed_size, p.hidden_size,1, p.enc_bidi,rnn_drop=p.enc_rnn_dropout).to(DEVICE)
    decoder = DecoderRNN(len(v), p.embed_size, p.dec_hidden_size,
                                  enc_attn=p.enc_attn, dec_attn=p.dec_attn,
                                  pointer=p.pointer,
                                  in_drop=p.dec_in_dropout, rnn_drop=p.dec_rnn_dropout,
                                  out_drop=p.dec_out_dropout, enc_hidden_size=p.hidden_size * 2).to(DEVICE)    
    sent_vec=[v[word] for word in sent.split()]
    vocab=deepcopy(v)
    for word in sent.split():
      vocab.add_words(word)
    sent_vec_extra=[vocab[word] for word in sent.split()] 
    print(len(sent_vec_extra)) 
    if(len(sent_vec_extra)<400):
        sent_vec_extra=F.pad(sent_vec_extra, 400-len(sent_vec_extra), 'constant')

    if(len(sent_vec)<400):
        sent_vec=F.pad(sent_vec, 400-len(sent_vec), 'constant')
    if(os.path.exists('/content/drive/My Drive/encoder_sum.pt')):
        encoder.load_state_dict(torch.load('/content/drive/My Drive/encoder_sum.pt',map_location=torch.device(DEVICE)))
        decoder.load_state_dict(torch.load('/content/drive/My Drive/decoder_sum.pt',map_location=torch.device(DEVICE)))
        enc_dec_adapter.load_state_dict(torch.load('/content/drive/My Drive/adapter_sum.pt',map_location=torch.device(DEVICE)))
        embedding.load_state_dict(torch.load('/content/drive/My Drive/embedding_sum.pt',map_location=torch.device(DEVICE)))
    x=torch.tensor(sent_vec).view(1,-1).to(DEVICE)
    x_extra=torch.tensor(sent_vec_extra).view(1,-1).to(DEVICE)
    encoder_embedded = embedding(x)
    encoder_outputs, encoder_hidden =encoder(encoder_embedded,torch.tensor(len(sent_vec)).view(1).to(DEVICE))
    decoder_input = torch.tensor([v.SOS] * batch_size, device=DEVICE)
    decoder_hidden = enc_dec_adapter(encoder_hidden)
    

    decoder_states = []
    enc_attn_weights = []
    # loss=0
    output=[]
    for di in range(100):
        decoder_embedded = embedding(decoder_input)
        if enc_attn_weights:
            coverage_vector = get_coverage_vector(enc_attn_weights)
        else:
            coverage_vector = None
        decoder_output, decoder_hidden, dec_enc_attn, dec_prob_ptr = decoder(decoder_embedded, decoder_hidden, encoder_outputs,
                    torch.cat(decoder_states) if decoder_states else None, coverage_vector,
                    encoder_word_idx=x_extra,log_prob=True,ext_vocab_size=len(vocab))  
        decoder_output.to(DEVICE)
        decoder_hidden.to(DEVICE)
        dec_enc_attn.to(DEVICE)
        dec_prob_ptr.to(DEVICE)

        decoder_states.append(decoder_hidden)      
        prob_distribution = torch.exp(decoder_output)# if log_prob else decoder_output
        # print(prob_distribution.shape)
        # top_idx = torch.multinomial(prob_distribution, 1)
        # top_idx = top_idx.squeeze(1).detach()  # detach from history as input
        _, top_idx = decoder_output.data.topk(1)
        output.append(top_idx.squeeze().data.item())
        enc_attn_weights.append(dec_enc_attn.unsqueeze(0)) 
        decoder_input = top_idx.view(-1)
    output=[vocab[idx] for idx in output]    
    return output 
    
       
if  __name__== "__main__:
    dataset = Dataset(p.data_path, max_src_len=p.max_src_len, max_tgt_len=p.max_tgt_len,
                        truncate_src=p.truncate_src, truncate_tgt=p.truncate_tgt)
    val_dataset=Dataset(p.val_data_path, max_src_len=p.max_src_len, max_tgt_len=p.max_tgt_len,
                        truncate_src=p.truncate_src, truncate_tgt=p.truncate_tgt)
    
    vocab = dataset.build_vocab(embed_file=p.embed_file)
    embedding_weights = torch.from_numpy(v.embeddings)    
    train(dataset,val_dataset,vocab,embedding_weights)
    
    #predict
    sent="( cnn student news ) -- april 30 , 2014 <P> another round of severe weather strikes the u.s. , concerns are voiced about whether rio de janeiro will be ready for the 2016 olympics , and a british cancer patient turns his bucket list into a list of ways he 's helped others . we also visit a mexican resort town , and we 'll show you what 's likely the cutest antelope you 've ever seen . <P> on this page you will find today 's show transcript , the daily curriculum , and a place for you to leave feedback . <P> transcript <P> click here to access the transcript of today 's cnn student news program . <P> please note that there may be a delay between the time when the video is available and when the transcript is published . <P> daily curriculum <P> click here for a printable version of the daily curriculum ( pdf ) . <P> media literacy question of the day : <P> what might be the pros and cons of getting vacation information from the destination 's tourism website ? where might you go for additional information ? <P> key concepts : identify or explain these subjects you heard about in today 's show : <P> 1 . international olympic committee ( ioc ) <P> 2 . tourism <P> 3 . `` bucket list '' <P> fast facts : how well were you listening to today 's program ? <P> 1 . about how many people in the u.s. were under the threat of severe weather yesterday ? what states were hit by suspected tornadoes ? what aspects of forecasting and communication are helping to save lives during tornado season ? what challenges to dealing with these storms still exist ? <P> 2 . what city is the site of the 2016 summer olympic games ? according to the head of the international olympic committee , is the city on track to be ready for this event ? explain . how has the city 's mayor responded to this claim ? <P> 3 . where is puerto vallarta located ? what are some of the challenges this resort has faced in recent years ? according to its tourism director , how was puerto vallarta 's tourism industry affected by these challenges ? what are some of the things the resort"
    summary=predict(sent,v)
    print(summary)