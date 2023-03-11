import torch 
import torch.nn as nn

import copy
###########################################################################
# biggest
class Transformer(nn.Module): 
    def __init__(self) -> None:
        super().__init__()

        
              
        pass 

###########################################################################
# medium 
class Encoder(nn.Module): 
    def __init__(self) -> None:
        super().__init__()

        pass
    
class Decoder(nn.Module): 
    def __init__(self) -> None:
        super().__init__()

        pass

class Generator(nn.Module):
    '''
    Linear + softmax
    '''
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
# small 
class EncoderBlock(nn.Module): 
    def __init__(self, seq_length, emb_size, hidden_size) -> None:
        super().__init__()
        self.key = nn.Linear(seq_length, emb_size)
        self.value = nn.Linear(seq_length, emb_size)
        self.query = nn.Linear(seq_length, emb_size)

        # self.attention =        
    def forward(self, x):
        '''
        x: (batch_size, seq_len, d_model)
        '''
        
        pass
        
class DecoderBlock(nn.Module): 
    def __init__(self) -> None:
        super().__init__()

###########################################################################
class Embeddings(nn.Module): 
    def __init__(self) -> None:
        super().__init__()
        
class PositionalEncoding(nn.Module): 
    def __init__(self) -> None:
        super().__init__()
        
class MultiHeadAttention(nn.Module): 
    def __init__(self) -> None:
        super().__init__()
        
class LayerNorm(nn.Module): 
    def __init__(self) -> None:
        super().__init__()
        
class PositionwiseFFN(nn.Module): 
    def __init__(self) -> None:
        super().__init__()
        
###########################################################################
def attention_head(query, key, value, mask=None): 
    '''
    Compute the scaled dot-product attention
    query: (batch_size, seq_len, d_model)
    
    '''
    
    d_k = key.size(-1) # same as query.size(-1)
    query_key = torch.matmul(query, key)
    if mask: 
        pass 
    scale = torch.sqrt(torch.tensor(d_k, dtype=torch.float))
    attention = query_key / scale
    attention = torch.softmax(attention, dim=-1)
    
    return torch.matmul(attention, value)
        
###########################################################################
# helper functions
def clones(module, N): 
    '''
    Produce N identical layers
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])