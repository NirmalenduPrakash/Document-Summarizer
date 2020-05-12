

import torch
import torch.nn as nn
import math
from torch import optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
# from typing import NamedTuple, List, Callable, Dict, Tuple, Optional
from collections import Counter
from random import shuffle
import numpy as np
import gzip
from torch.utils.data import Dataset, DataLoader
import pickle
from torch.nn.utils.rnn import pack_sequence
from torch.nn import functional as F

class Params:
  hidden_size = 512  
  shuffle=True
  learning_rate=.001
  num_epochs=100
  data_path = '/home/svu/e0401988/NLP/summarization/cnndm.gz'
  val_data_path = '/home/svu/e0401988/NLP/summarization/cnndm.val.gz'
  eps=1e-31
  batch_size=4
  model_path='/home/svu/e0401988/NLP/summarization/model.pt'
  # decoder_weights_path='/content/drive/My Drive/decoder.pt'
  losses_path='/content/drive/My Drive/val_losses.pkl'

  # Testing
  test_data_path: str = '/home/svu/e0401988/NLP/summarization/cnndm.test.gz'

def count_words(count_dict, text):
    '''Count the number of occurrences of each word in a set of text'''
    for sentence in text:
        for word in sentence:
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1

def simple_tokenizer(text, lower=False, newline=None):
  if lower:
    text = text.lower()
  if newline is not None:  # replace newline by a token
    text = text.replace('\n', ' ' + newline + ' ')
  return text.split()

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
    
  def __getitem__(self, item):
    if type(item) is int:
      return self.index2word[item]
    return self.word2index.get(item, self.UNK)

  def __len__(self):
    return len(self.index2word)

class Dataset(nn.Module):
  def __init__(self, filename, tokenize=simple_tokenizer):
    print("Reading dataset %s..." % filename, end=' ', flush=True)
    self.filename = filename
    self.pairs = []
    self.src_len = 0
    self.tgt_len = 0
    if filename.endswith('.gz'):
      open = gzip.open
    with open(filename, 'rt', encoding='utf-8') as f:
      for i, line in enumerate(f):
        pair = line.strip().split('\t')
        if len(pair) != 2:
          print("Line %d of %s is malformed." % (i, filename))
          continue
        src = tokenize(pair[0])
        tgt = tokenize(pair[1])
        self.pairs.append((src, tgt))
    print("%d pairs." % len(self.pairs))

  def build_vocab(self):
    # word frequency
    word_counts={}
    count_words(word_counts,[src+tgr for src,tgr in self.pairs])
    vocab=Vocab()
    for word,count in word_counts.items():
        if(count>20):
            vocab.add_words([word])  
    self.vocab=vocab 
  def vectorize(self,tokens):
    return [self.vocab[token] for token in tokens]
  def unvectorize(self, indices):
    return [self.vocab[i] for i in indices]
  def __getitem__(self, index):
    return {'x':self.vectorize(self.pairs[index][0]),
            'y':self.vectorize(self.pairs[index][1]),
            'src':self.pairs[index][0],
            'trg':self.pairs[index][1],
            'x_len':len(self.pairs[index][0]),
            'y_len':len(self.pairs[index][1])}
  def __len__(self):
    return len(self.pairs)

"""Positional Embedding
*   Sin and cos functions represent binary alternating sequence
*   Both Sin and cos are used as it allows for linear transformation, and hence relative position, which varies linearly with time
*   As mentioned in "attention is all you need" paper, learn embeddings are almost identical to sinusoidal embeddings, i decide to go with fixed sinusoidal embeddings

query
*   Why not just use One-hot embedding instead of sin,cos functions (to be experimented)
*   Learnt embedding is made larger by multiplying with square root of embedding dimension (as mentioned in paper); i suppose word meaning is more important than it's position in sequence and this larger number makes training faster. To be removed and experimented. Also, why not just use a constant to multiply?
"""

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 3000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model,requires_grad=False)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        self.pe = pe.unsqueeze_(0)
        # self.register_buffer('pe', pe)
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add positional embedding
        seq_len = x.size(1)
        # pe = torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        # if x.is_cuda:
        #     pe.cuda()
        pe = torch.tensor(self.pe[:,:seq_len],requires_grad=False).to(device)
        x = x + pe
        return x

"""*   From the paper - 'We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients.To counteract this effect, we scale the dot products'
Softmax function:
<img src="https://drive.google.com/uc?id=1qVvoGO6CPj7-tFYDHmrZNh2OGOG92WTV" alt="Drawing" Width=300 Height=250/><img src="https://drive.google.com/uc?id=1qQJedobeXv1YyrpvYkgZLnOJUczXk5qs" alt="Drawing" Width=300 Height=250/>

If the magnitudes grow large, the softmax function is such that it magnifies such values and suppress lower values, in terms of probabilities.
On the other hand, if we reduce the magnitude, we get a gradual slope and for most inputs, we have bigger gradients.

*   Masking is used for decoder self attention, because decoding is done sequentially and during training we consider the whole sequence as it makes training faster. To factor time step, I assign 0 to leaked attentions.
Suppose decoded sequence is 5 tokens long. Scores last 2 dimensions will be 5x5.Here rows represent time steps, hence we need to make non-highlighted cells 0. 

  ![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAUYAAABRCAYAAACucw9nAAAEn0lEQVR4Ae3dW5IbMQxDUe9/y/nwRA+y29wCjqs6sj7FAi4htUf5fD6fr0cNaIAGaOBHA5/vv2/2cwXxCf+sBpn+UYMTlOhgQQEYd6cghvQKgMJSgOZwawCMxMAQpymAAi+0F4CRGFoMhw+x/wIjL7QXgJEYWgyxSDwLB0ZeaC8AIzG0GIAxvAK80F4ARmJoMYRjQWLkhfYCMBJDiwEYwyvAC+0FYCSGFkM4FiRGXmgvACMxtBiAMbwCvNBeAEZiaDGEY0Fi5IX2AjASQ4sBGMMrwAvtBWAkhhZDOBYkRl5oLwAjMbQYgDG8ArzQXgBGYmgxhGNBYuSF9sISg0cNaIAGaODRgGvHXL3WjTE8MLpyy7Vj2wKrQQCjGgDj7QjbEOHdQQ1OagRGYARGYOx2AIzAeJoCMAIjMAJjVwAYgbH+W4vVHF7CSP2qBnSwtL914Oc6zlmXEEBBDRoKqZ3xrhsYV2KylT41CDcDKBwBaJASo620rfRPOwAFqbkbpK20rfROzT+IyJwAIzAC4zst1ffU0Rnj7gTACIzAWBB0xuiM8QZkYARGYATG58f9EqPEqDncCtzm4IzRGaMzxuMJiVFiXErYOgBGYARGYKy4pDkAo5/rvI8TyhnBIyhIjEv+JzHeL5UajKcw6qAONJCrgecAvtJD2rgaQ9qa53rVwJv5u1NYzSD9s2sACsC4k9GEZdq8tlDhVADG2kqnGWCuV1o6aWnWJW0OjLslACMwevlS8NMcbKVvUgZGYARGYHzOlyVGifHdHJwxOmN0xnivn7vGSB4kRolRYpQYJcbRBYARGIERGIERGEcFgBEYgREYBxYkRmAERmAERmAcFQBGYARGYBxYkBiBERiBERiBcVQAGIERGIFxYEFiBEZgBEZgBMZRAWAERmAExoEFibHAeP8UahXEowY0QAM04PIAN8us1EgHalA6GAkycbr88GwlaluVNqoBKBQU0rQ/17u8kEjCsWZgZAjnrAUHDfI0yAGJxCkwAiMwAuOza5QYdx8ARmAERmAExhGLgREYgREYgREYv48IGOKphfM1L6AqJAxIJE4lxhJDQTJ1BEZgLC8kknCsGRhLDKlArHUDIzCWFwYkEqfAWGIoQKSOwAiM5YVEEo41A2OJIRWItW5gBMbywoBE4hQYSwwFiNQRGIGxvJBIwrFmYCwxpAKx1g2MwFheGJBInAJjiaEAkToCIzCWFxJJONYMjCWGVCDWuoERGMsLAxKJ0wPGZQqPGtAADdBAa+D564dKD2njEkPamud61eAYYtYlbU4HwNgwJAZQWACkAzUoHazzgwZEWnes9aoBQ5QhShOpIy8cLwCjpLAbI0NoDprDCYnLC8AIjMB4b1zSHDSHag7ACIzACIx9nKY52EoTw+tuSoaQliotpZ6v1rptpSUFzUFzaA0AozNGYngBgSEeQ1RqSB3tHGylG47EYBupOWgO1QxtpW2lNYdXctYgNchqkN5Keyu94QgKoFBQqOSUOkqMEqPEKDG2BoDxOU6QGCVGiVGDbDjaOXj5QgzSUmtAWnrSUuoWutbdW+n7xZ2M7qWkARqggf8a+ANXp50r5BOrOAAAAABJRU5ErkJggg==)
"""

def attention(q, k, v, d_k,mask=False):
  scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(d_k)
  scores = F.softmax(scores, dim=-1) 
  tensor_mask=torch.ones(scores.shape).to(device)  
  if mask:
    for row in range(tensor_mask.size(-2)):
      for col in range(tensor_mask.size(-1)):
        if(row>col):
          tensor_mask[:,:,row,col]=0 
    scores=scores * tensor_mask       
    scores=scores.transpose(-2,-1)           
  output = torch.matmul(scores, v)
  return output

"""*   From the paper - "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this." - Not clear, to be experimented.
*   In case of self attention, q,k,v are encoder or decoder embeddings. Perhaps, we want to learn 2 representations for each token - q and k (similar to word2vec where we have 2 vectors for each word, one when it is part of context and when it is central word?). This would explain multiple heads and then linear transformation(to be verified).
"""

class MultiHeadAttention(nn.Module):
    def __init__(self,hidden_size,heads=8):
        super(MultiHeadAttention, self).__init__()
        self.heads=heads 
        self.d_k = hidden_size // heads
        self.hidden_size=hidden_size
        self.q_linear = nn.Linear(hidden_size, hidden_size,bias=False)
        self.v_linear = nn.Linear(hidden_size, hidden_size,bias=False)
        self.k_linear = nn.Linear(hidden_size, hidden_size,bias=False)        
    
    def forward(self, q, k, v,mask=False):        
        bs = q.size(0)
        k = self.k_linear(k).view(bs,-1,self.heads,self.d_k)
        q = self.q_linear(q).view(bs,-1,self.heads,self.d_k)
        v = self.v_linear(v).view(bs,-1,self.heads,self.d_k)

        k=k.transpose(1,2)
        q=q.transpose(1,2)
        v=v.transpose(1,2)

        scores = attention(q, k, v, self.d_k,mask)
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.hidden_size)
        return concat
        # return scores.view(bs, -1, self.hidden_size)

"""From pytorch documentation - "γ and β are learnable affine transform parameters of normalized_shape if elementwise_affine is True."
-- To be explored
"""

class Norm(nn.Module):
  def __init__(self,hidden_size):
    super(Norm, self).__init__()
    self.norm=nn.LayerNorm(hidden_size,elementwise_affine=False)
  def forward(self,x,res):
    x=x+self.norm(res)
    return x

class FeedForward(nn.Module):
  def __init__(self,hidden_size,d_ff=2048):
    super(FeedForward, self).__init__()
    self.linear_1=nn.Linear(hidden_size,d_ff)
    self.linear_2=nn.Linear(d_ff,hidden_size)
  def forward(self,x):
    output=self.linear_1(x)
    output=self.linear_2(output)
    return output

class Encoder(nn.Module):
  def __init__(self,hidden_size,vocab_size):
    super(Encoder, self).__init__()
    self.hidden_size=hidden_size
    self.pos=PositionalEncoder(hidden_size)
    self.embedding=nn.Embedding(vocab_size,hidden_size,padding_idx=0)
    self.attn=MultiHeadAttention(hidden_size)
    self.norm=Norm(hidden_size)
    self.ff=FeedForward(hidden_size)
  def forward(self,seq):
    embedded=self.embedding(seq)
    embedded+=self.pos(embedded)
    output=self.attn(embedded,embedded,embedded)
    output=self.norm(embedded,output)
    res=self.ff(output)
    output=self.norm(output,res)
    return output

class Decoder(nn.Module):
  def __init__(self,hidden_size,vocab_size):
    super(Decoder, self).__init__()
    self.hidden_size=hidden_size
    self.pos=PositionalEncoder(hidden_size)
    self.embedding=nn.Embedding(vocab_size,hidden_size,padding_idx=0)
    self.attn_1=MultiHeadAttention(hidden_size)
    self.attn_2=MultiHeadAttention(hidden_size)
    self.norm=Norm(hidden_size)
    self.ff=FeedForward(hidden_size)
  def forward(self,seq,enc_output):
    embedded=self.embedding(seq)
    embedded+=self.pos(embedded)
    output=self.attn_1(embedded,embedded,embedded,True)
    output=self.norm(embedded,output)
    res=self.attn_2(output,enc_output,enc_output)
    output=self.norm(output,res)
    res=self.ff(output)
    dec_output=self.norm(output,res)
    return output,dec_output,embedded

class Transformer(nn.Module):
  def __init__(self,vocab,hidden_size):
    super(Transformer, self).__init__()
    self.vocab=vocab
    self.vocab_size=len(vocab)
    self.hidden_size=hidden_size
    self.encoder=Encoder(hidden_size,self.vocab_size)
    self.decoder=Decoder(hidden_size,self.vocab_size)
    self.linear=nn.Linear(hidden_size,self.vocab_size)
    self.softmax=nn.Softmax()
    self.sigmoid=nn.Sigmoid()
    self.attn=MultiHeadAttention(hidden_size) 
    self.enc_linear=nn.Linear(hidden_size,1)
    self.dec_linear=nn.Linear(hidden_size,1)
    self.embd_linear=nn.Linear(hidden_size,1)

  def forward(self,x,y,extra_vocab,expanded_word_idx):
    enc_output=self.encoder(x)
    output,dec_output,embedded=self.decoder(y,enc_output)
    gen_output=self.softmax(self.linear(dec_output))
    ptr_output=self.attn(enc_output,output,output) 

    ptr_mean=torch.mean(ptr_output,-2).squeeze(-1)
    pb_gen=[]
    for b in range(dec_output.size(0)):
      p_gen=[]
      for t in range(dec_output.size(-2)):
        p_gen.append(self.sigmoid(self.enc_linear(ptr_mean[b,:])+self.dec_linear(dec_output[b,t,:])+self.embd_linear(embedded[b,t,:])))
      pb_gen.append(p_gen)  
    pb_gen=torch.tensor(pb_gen).to(device)
    gen_padded=torch.cat((gen_output,torch.zeros(gen_output.size(0),gen_output.size(1),len(extra_vocab),device=device)),2)
        
    ptr_dist=F.softmax(torch.matmul(output, enc_output.transpose(-2,-1)),dim=-1)  
    pb_ptr=(1-pb_gen).unsqueeze(-1).expand_as(ptr_dist)
    gen_padded.scatter_add_(2,expanded_word_idx.unsqueeze(1).expand(-1,gen_padded.size(1),-1),ptr_dist*pb_ptr)
    return torch.log(gen_padded+1e-5)

def my_collate(batch):
    max_x=np.max([item['x_len'] for item in batch])
    max_y=np.max([item['y_len'] for item in batch])
    
    x = [F.pad(torch.tensor(item['x']), (0,max_x-len(item['x'])), 'constant').tolist() for item in batch]    
    y = [F.pad(torch.tensor(item['y']), (0,max_y-len(item['y'])), 'constant').tolist() for item in batch]
    
    src=[item['src'] for item in batch]
    trg=[item['trg'] for item in batch]
    return {'x':x,'y': y,'src':src,'trg':trg,'x_len':[item['x_len'] for item in batch],'y_len':[item['y_len'] for item in batch]}

with torch.autograd.set_detect_anomaly(True):
  p=Params()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # device="cpu"
  batch_size=p.batch_size
  dataset=Dataset(p.data_path)
  dataset.build_vocab()
  valset=Dataset(p.val_data_path)
  valset.vocab=dataset.vocab
  dataloader = DataLoader(dataset,p.batch_size,p.shuffle,collate_fn=my_collate)
  valloader = DataLoader(valset,p.batch_size,p.shuffle,collate_fn=my_collate)
  model=Transformer(dataset.vocab,p.hidden_size).to(device)
  optimizer = optim.Adam(model.parameters(), lr=p.learning_rate)
  # criterion = nn.NLLLoss(ignore_index=dataset.vocab.PAD)

  training_loss=[]
  validation_loss=[]
  for _e in range(p.num_epochs):
    epoch_tr_loss=0
    epoch_vl_loss=0
    for batch in tqdm(dataloader):
      # print(torch.cuda.memory_allocated(device="cuda"))
      optimizer.zero_grad()
      extra_vocab={}
      expanded_x=[]
      expanded_y=[]
      
      x=torch.tensor(batch['x']).to(device)
      y=torch.tensor(batch['y']).to(device)
      src=batch['src']
      trg=batch['trg']

      max_src=np.max(batch['x_len'])
      for sent in src:
        vec=[]
        for token in sent:
          if(token in extra_vocab.keys()):
            vec.append(extra_vocab[token]) 
          elif(token not in dataset.vocab.word2index):
            extra_vocab[token]=len(dataset.vocab)+len(extra_vocab)
            vec.append(extra_vocab[token])
        expanded_x.append(vec + [0 for i in range(max_src-len(vec))])
      expanded_x=torch.tensor(expanded_x).to(device)  

      max_trg=np.max(batch['y_len'])
      for sent in trg:
        vec=[]
        for token in sent:
          if(token in extra_vocab.keys()):
            vec.append(extra_vocab[token]) 
          elif(token not in dataset.vocab.word2index):
            extra_vocab[token]=len(dataset.vocab)+len(extra_vocab)
            vec.append(extra_vocab[token])
        expanded_y.append(vec + [0 for i in range(max_trg-len(vec))])          
      expanded_y = torch.tensor(expanded_y).to(device)
      
      y=torch.cat((torch.tensor([dataset.vocab.SOS] * y.size(0), device=device).view(-1,1),y[:,:-1]),dim=1)
      pred=model(x,y,extra_vocab,expanded_x)
      loss=F.nll_loss(pred.permute(0,2,1), y, ignore_index=dataset.vocab.PAD, reduction='sum')
      loss.backward()
      optimizer.step() 
      epoch_tr_loss+=loss.data.item()
    training_loss.append(epoch_tr_loss)
    for batch in tqdm(valloader):
      optimizer.zero_grad()
      extra_vocab={}
      expanded_x=[]
      expanded_y=[]
      
      x=torch.tensor(batch['x']).to(device)
      y=torch.tensor(batch['y']).to(device)
      src=batch['src']
      trg=batch['trg']

      max_src=np.max(batch['x_len'])
      for sent in src:
        vec=[]
        for token in sent:
          if(token in extra_vocab.keys()):
            vec.append(extra_vocab[token]) 
          elif(token not in dataset.vocab.word2index):
            extra_vocab[token]=len(dataset.vocab)+len(extra_vocab)
            vec.append(extra_vocab[token])
        expanded_x.append(vec + [0 for i in range(max_src-len(vec))])
      expanded_x=torch.tensor(expanded_x).to(device)  

      max_trg=np.max(batch['y_len'])
      for sent in trg:
        vec=[]
        for token in sent:
          if(token in extra_vocab.keys()):
            vec.append(extra_vocab[token]) 
          elif(token not in dataset.vocab.word2index):
            extra_vocab[token]=len(dataset.vocab)+len(extra_vocab)
            vec.append(extra_vocab[token])
        expanded_y.append(vec + [0 for i in range(max_trg-len(vec))])          
      expanded_y = torch.tensor(expanded_y).to(device)
      
      y=torch.cat((torch.tensor([dataset.vocab.SOS] * y.size(0), device=device).view(-1,1),y[:,:-1]),dim=1)
      pred=model(x,y,extra_vocab,expanded_x)
      loss=F.nll_loss(pred.permute(0,2,1), y, ignore_index=dataset.vocab.PAD, reduction='sum')
      loss.backward()
      optimizer.step() 
      epoch_vl_loss+=loss.data.item()
    print(epoch_vl_loss)  
    validation_loss.append(epoch_vl_loss)
    if(len(validation_loss)>0 and epoch_vl_loss<min(validation_loss)):
      torch.save(model.state_dict(), p.model_path)
