
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    '''
    Multi-Head Attention mechanism for transformer models.

    Arguments:
        dmodel: Dimension of the model
        heads: Number of attention heads

    Methods:
        forward(x): Perform multi-head attention on the input tensor x
    '''
    def __init__(self, dmodel, heads):
        super(MultiHeadAttention, self).__init__()

        self.dmodel = dmodel
        self.heads = heads
        self.head_size = dmodel // heads

        self.k_linear = nn.Linear(dmodel, dmodel)
        self.q_linear = nn.Linear(dmodel, dmodel)
        self.v_linear = nn.Linear(dmodel, dmodel)
        self.out_linear = nn.Linear(dmodel, dmodel)

    def split_heads(self, x, batch_size):
        '''
        Split the last dimension into (heads, head_size) and transpose to shape (batch_size, heads, seq_len, head_size).
        '''
        return x.view(batch_size, -1, self.heads, self.head_size).transpose(1, 2)

    def attention(self, k, q, v):
        '''
        Compute the attention weights and apply them to the value vectors.
        '''
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32, device=q.device))
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    def forward(self, x):
        '''
        Perform the multi-head attention mechanism on the input tensor x.
        '''
        batch_size = x.size(0)

        K = self.split_heads(self.k_linear(x), batch_size)  # Key: What can I offer
        Q = self.split_heads(self.q_linear(x), batch_size)  # Query: What am I looking for
        V = self.split_heads(self.v_linear(x), batch_size)  # Value: What I actually offer

        attn_output = self.attention(K, Q, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.dmodel)
        
        return self.out_linear(attn_output)
