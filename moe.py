import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.nn.modules.sparse import Embedding
import math

tokenizer = AutoTokenizer.from_pretrained("/Users/mithunnayak/Desktop/WORK/MOE/gpt_tokenizer")
# tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl", download=True)
# tokenizer.save_pretrained("./gpt_tokenizer")
vocabulary_size = tokenizer.vocab_size


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(self.pe[:, :x.size(1)].shape)
        return x + self.pe[:, :x.size(1)]




class GPT_Transformer(nn.Module):
    def __init__(self, model_dim, vocab_size, blocks):
        super().__init__()
        self.blocks = blocks
        self.pos_encod = PositionalEncoding(d_model=model_dim, max_len=1000)
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        self.embeddings = Embedding(vocab_size, model_dim)
        self.mha = torch.nn.MultiheadAttention(model_dim, 12,batch_first=True)
        self.linear1 = torch.nn.Linear(model_dim, model_dim * 4)
        self.linear2 = torch.nn.Linear(model_dim * 4, model_dim)
        self.relu = torch.nn.ReLU()
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.llm_head = nn.Linear(self.model_dim, self.vocab_size)


    def generate_causal_mask(self,seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf')).float()
        return mask

    def forward(self, x):
        tokens = torch.tensor(tokenizer(x)['input_ids']).unsqueeze(0)
        embed = self.embeddings(tokens)
        positional_encoding_embeddings = self.pos_encod(embed)
        x_in = positional_encoding_embeddings
        seq_len = x_in.size(1)
        mask = self.generate_causal_mask(seq_len)
        for i in range(self.blocks):
            x_in = self.norm1(x_in)
            attn_output, attn_weights  = self.mha(x_in,x_in,x_in,attn_mask=mask)
            #Mixture - of - Experts layer for FFN

        
        logits = self.llm_head(x_in)
        prob = torch.softmax(logits, dim=-1)
        next_token = torch.argmax(prob, dim=-1)
        return next_token 










