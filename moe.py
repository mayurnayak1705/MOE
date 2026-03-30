import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.nn.modules.sparse import Embedding
import math

tokenizer = AutoTokenizer.from_pretrained("./gpt_tokenizer")
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
        # print(out_noise)
        noise = torch.rand_like(out_noise)
        noise_add = out_noise * noise
        out_1 = out_1 + noise_add
        #end of noise generation - addition part

        max_k, max_k_index = torch.topk(out_1, k=self.top_k, dim=-1) # find the top 2 highest percentage experts (top 2 values, thier index)
        gating = torch.full_like(out_1, float('-inf')) # create a 0 tensor of sf type
        sparse_top_k = gating.scatter(-1,max_k_index, max_k) #fill that tensor with -inf's everywhere except the top-2 positions of experts (keep thier values same)
        sf_sparse_topk = torch.nn.functional.softmax(sparse_top_k, dim=-1)
        return sf_sparse_topk, max_k_index #(batch, seq, emd)
    
# num_experts = 4
# top_k = 2
# n_embd = 32
x = torch.rand(2, 10, 32)
# model = Topk_noisy_Router(d_model=n_embd, num_of_experts=num_experts, top_k=top_k)
# top_k_matrix, index = model(x)
# print(top_k_matrix)
# print(index)


class Expert(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.layer1 = nn.Linear(d_model, 4*d_model)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(4*d_model, d_model)
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, x):
        out = self.dropout(self.layer2(self.relu(self.layer1(x))))
        return out


class SparseMOE(nn.Module):
    def __init__(self, d_model, num_of_experts, top_k):
        super().__init__()
        self.d_model = d_model
        self.num_of_experts = num_of_experts
        self.topk = top_k
        self.router = Topk_noisy_Router(d_model=self.d_model, num_of_experts=self.num_of_experts, top_k=self.topk)
        self.experts = nn.ModuleList([Expert(self.d_model) for _ in range(num_of_experts)])

    def forward(self, x):
        gating_output, index = self.router(x)
        final_output = torch.zeros_like(x)

        flat_x = x.view(-1, x.size(-1))
        # print(x.shape) #batch, seq, dim
        # print(flat_x.shape) # batch * seq , dim
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))
        # print(gating_output.shape) #batch,seq, experts
        # print(flat_gating_output.shape) #batch*seq, experts
        for i, expert in enumerate(self.experts):
            expert_mask = (index== i).any(dim=-1) #checks if any token is assigned to the expert 'i'
            flat_mask = expert_mask.view(-1) #shaping it to batch*seq , 
            if flat_mask.any():
                expert_input = flat_x[flat_mask] #get the seq tokens who has expert-i to be processed
                logits = expert(expert_input)
                # print([flat_mask, i])
                # print(flat_gating_output.shape)
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = logits * gating_scores
                final_output[expert_mask] += weighted_output.squeeze(1)
                # print(expert_mask)
        return final_output



# model = SparseMOE(32, 10, 2)
# print(model(x).shape)


class MOE_Transformer(nn.Module):
    def __init__(self, model_dim, vocab_size, head, blocks, num_of_experts, topk):
        super().__init__()
        self.blocks = blocks
        self.head = head
        self.num_of_experts = num_of_experts
        self.topk = topk
        self.pos_encod = PositionalEncoding(d_model=model_dim, max_len=1000)
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        self.embeddings = Embedding(vocab_size, model_dim)
        self.mha = torch.nn.MultiheadAttention(model_dim, self.head,batch_first=True)
        self.linear1 = torch.nn.Linear(model_dim, model_dim * 4)
        self.linear2 = torch.nn.Linear(model_dim * 4, model_dim)
        self.relu = torch.nn.ReLU()
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.llm_head = nn.Linear(self.model_dim, self.vocab_size)
        self.moe_layer =SparseMOE(self.model_dim,self.num_of_experts,self.topk)


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
            x_in = self.norm1(x_in + attn_output)
            ff = self.moe_layer(x_in)
            # print(f"shape after ffn {ff.shape}")
            x_in = self.norm2(x_in + ff)
        logits = self.llm_head(x_in)
        prob = torch.softmax(logits, dim=-1)
        next_token = torch.argmax(prob, dim=-1)
        return next_token 


model = MOE_Transformer(model_dim=32,vocab_size=vocabulary_size,head=8,blocks=8,num_of_experts=10,topk=2)
print(model("Hi, my name"))





