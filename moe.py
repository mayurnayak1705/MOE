import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.nn.modules.sparse import Embedding
import math

tokenizer = AutoTokenizer.from_pretrained("/Users/mithunnayak/Desktop/WORK/MOE/gpt_tokenizer")
# tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl", download=True)
# tokenizer.save_pretrained("./gpt_tokenizer")
vocabulary_size = tokenizer.vocab_size

torch.manual_seed(1445)

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


class Topk_Router(nn.Module):
    def __init__(self, d_model, num_of_experts,top_k):
        super().__init__()
        self.d_model = d_model
        self.num_of_experts = num_of_experts
        self.top_k = top_k
        self.router = torch.nn.Linear(d_model, num_of_experts)

    def forward(self, x):
        out_1 = self.router(x) #get the token projected to experts
        max_k, max_k_index = torch.topk(out_1, k=self.top_k, dim=-1) # find the top 2 highest percentage experts (top 2 values, thier index)
        gating = torch.full_like(out_1, float('-inf')) # create a 0 tensor of sf type
        sparse_top_k = gating.scatter(-1,max_k_index, max_k) #fill that tensor with -inf's everywhere except the top-2 positions of experts (keep thier values same)
        sf_sparse_topk = torch.nn.functional.softmax(sparse_top_k, dim=-1)
        return sf_sparse_topk, max_k_index #(batch, seq, emd)
    


class Topk_noisy_Router(nn.Module):
    def __init__(self, d_model, num_of_experts,top_k):
        super().__init__()
        self.d_model = d_model
        self.num_of_experts = num_of_experts
        self.top_k = top_k
        self.router = torch.nn.Linear(d_model, num_of_experts)
        self.noise = torch.nn.Linear(d_model, num_of_experts)


    def forward(self, x):
        out_1 = self.router(x) #get the token projected to experts

        #generating noise and adding to the above output for the load balancing
        out_noise = self.noise(x)
        out_noise = torch.nn.functional.softplus(out_noise)
        print(out_noise)
        noise = torch.rand_like(out_noise)
        noise_add = out_noise * noise
        out_1 = out_1 + noise_add
        #end of noise generation part
        
        max_k, max_k_index = torch.topk(out_1, k=self.top_k, dim=-1) # find the top 2 highest percentage experts (top 2 values, thier index)
        gating = torch.full_like(out_1, float('-inf')) # create a 0 tensor of sf type
        sparse_top_k = gating.scatter(-1,max_k_index, max_k) #fill that tensor with -inf's everywhere except the top-2 positions of experts (keep thier values same)
        sf_sparse_topk = torch.nn.functional.softmax(sparse_top_k, dim=-1)
        return sf_sparse_topk, max_k_index #(batch, seq, emd)
    
num_experts = 4
top_k = 2
n_embd = 32
x = torch.rand(1, 10, 32)
model = Topk_noisy_Router(d_model=n_embd, num_of_experts=num_experts, top_k=top_k)
top_k_matrix, index = model(x)
# print(top_k_matrix)
# print(index)






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










