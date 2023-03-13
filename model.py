import torch 
import torch.nn as nn
import math

import copy

###########################################################################
# The Transformer Model
class Transformer(nn.Module): 

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

###########################################################################
# Encoder, Decoder, Generator
class Encoder(nn.Module):
    def __init__(self, layer, N) -> None:
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers: 
            x = layer(x, mask)
        return self.norm(x)
    
class EncoderBlock(nn.Module): 

    def __init__(self, size, self_attn, feed_forward, dropout) -> None:
        super(EncoderBlock, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
        
class Decoder(nn.Module): 
    def __init__(self, layer, N) -> None:
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        '''
        '''
        for layer in self.layers: 
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # initialize sublayers 3 times
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        # in the forward function, we pass x and a function to the sublayer
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class Generator(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        '''
        x: (batch_size, seq_len, d_model)
        '''
        x = self.linear(x)
        x = self.softmax(x)
        
        return x
    
###########################################################################
class Embeddings(nn.Module): 
    def __init__(self, d_model, vocab_size) -> None:
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
        
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
    
# class MultiHeadedAttention(nn.Module):
#     def __init__(self, h, d_model, dropout=0.1):
#         "Take in model size and number of heads."
#         super(MultiHeadedAttention, self).__init__()
#         assert d_model % h == 0
#         # We assume d_v always equals d_k
#         self.d_k = d_model // h
#         self.h = h
#         self.linears = clones(nn.Linear(d_model, d_model), 4)
#         self.attn = None
#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, query, key, value, mask=None):
#         "Implements Figure 2"
#         if mask is not None:
#             # Same mask applied to all h heads.
#             mask = mask.unsqueeze(1)
#         nbatches = query.size(0)

#         # 1) Do all the linear projections in batch from d_model => h x d_k
#         query, key, value = [
#             lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
#             for lin, x in zip(self.linears, (query, key, value))
#         ]

#         # 2) Apply attention on all the projected vectors in batch.
#         x, self.attn = attention(
#             query, key, value, mask=mask, dropout=self.dropout
#         )

#         # 3) "Concat" using a view and apply a final linear.
#         x = (
#             x.transpose(1, 2)
#             .contiguous()
#             .view(nbatches, -1, self.h * self.d_k)
#         )
#         del query
#         del key
#         del value
#         return self.linears[-1](x)
        
class MultiHeadedAttention(nn.Module): 

    def __init__(self, h, d_model, dropout=0.1) -> None:
        '''
        h: number of heads
        d_model: dimension of model
        dropout: dropout probability
        
        '''
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        
        # We assume d_v always equals d_k
        # d_k = d_v = d_model // h, is the hidden dimension of each head
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4) # W^Q, W^K, W^V, W^O
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
        
    def forward(self, query, key, value, mask=None):
        '''
        query: (batch_size, seq_len, d_model)
        '''
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        
        batch_size = query.size(0)
        
        # step 1: input linear transformation
        # Apply a linear transformation to the query, key, and value tensors using the corresponding linear layers in self.linears
        query_linear = self.linears[0](query)
        key_linear = self.linears[1](key)
        value_linear = self.linears[2](value)
        
        # note: d_model = num_heads * head_dim
        # before reshaping: (batch_size, d_model, num_heads, head_dim)
        # Reshape the output tensors into 4D tensors with dimensions (batch_size, num_heads, d_model, head_dim)
        query_4d = query_linear.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key_4d = key_linear.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value_4d = value_linear.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        
        # step 2: apply attention
        x = self.attention_head(query_4d, key_4d, value_4d, mask, dropout=self.dropout)
        
        # step 3: concatenate the output of the attention heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        del query
        del key
        del value
        return self.linears[-1](x) # W^O
        
        
    def attention_head(self, query, key, value, mask=None, dropout=None): 
        '''
        Compute the scaled dot-product attention
        query: (batch_size, num_heads, d_model, head_dim)
        key: (batch_size, num_heads, d_model, head_dim)
        value: (batch_size, num_heads, d_model, head_dim)
        '''
        
        d_k = query.size(-1) # same as query.size(-1)
        query_key = torch.matmul(query, key.transpose(-2, -1))
        if mask is not None: 
            # mask = mask.unsqueeze(1)
            query_key = query_key.masked_fill(mask == 0, -1e9)
             
        scale = torch.sqrt(torch.tensor(d_k, dtype=torch.float))
        attention = query_key / scale
        # attention scores, shape: (batch_size, input_len, input_len)
        attention = torch.softmax(attention, dim=-1)
        if dropout is not None:
            attention = dropout(attention)
            
        # value vectors, shape: (batch_size, input_len, d_model)
        return torch.matmul(attention, value)        
        
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1) -> None:
        super(PositionwiseFeedForward, self).__init__()
        
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x): 
        return self.w_2(self.dropout(self.w_1(x).relu()))
        
###########################################################################
# helper functions/classes and non-parameterized layers
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1) -> None:
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
        
class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6) -> None:
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        
def clones(module, N): 
    '''
    Produce N identical layers
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])