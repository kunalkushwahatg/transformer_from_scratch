
import torch.nn as nn
from model.PositionalEncoding import PositionalEncoding
from model.MultiHeadedAttention import MultiHeadAttention
from model.FeedForward import FeedForward
from model.AddAndNorm import AddAndNorm



from config import Config

device = Config.DEVICE

class Encoder(nn.Module):
    '''
    Transformer Encoder implementation.

    Arguments:
        vocab_size: Size of the vocabulary
        shape: Shape of the input tensor (batch_size, max_len, dmodel)
        heads: Number of attention heads

    Methods:
        forward(x): Forward pass through the encoder
    '''
    def __init__(self, vocab_size,  shape,device, heads=4):
        super(Encoder, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, shape[2])
        self.positional_encoding = PositionalEncoding(shape,device=device)
        self.multi_headed_attention = MultiHeadAttention(shape[2], heads)
        self.add_and_norm1 = AddAndNorm(shape[2])
        self.feed_forward = FeedForward(dmodel=shape[2])
        self.add_and_norm2 = AddAndNorm(shape[2])
        self.linear = nn.Linear(shape[2], 512)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.token_embedding_table(x)
        residual = self.positional_encoding(out)
        out = self.multi_headed_attention(residual)
        
        residual = self.add_and_norm1(out, residual)
         
        out = self.feed_forward(residual)
        out = self.add_and_norm2(out, residual)
        
        out = self.linear(out)
        
        return out
