
import torch.nn as nn
from model.Encoder import Encoder

class Pretraining(nn.Module):
    '''
    Pretraining model for next word prediction using a transformer encoder.

    Arguments:
        vocab_size: Size of the vocabulary
        shape: Shape of the input tensor (batch_size, max_len, dmodel)
        heads: Number of attention heads

    Methods:
        forward(x): Forward pass through the pretraining model
        predict_next_word(x): Predict the next word for the input sequence
    '''
    def __init__(self, vocab_size, shape, deivce, heads=4):
        super(Pretraining, self).__init__()
        self.encoder = Encoder(vocab_size, shape, deivce,heads)
        self.linear = nn.Linear(shape[2] * shape[1], vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.encoder(x)
        out = out.view(out.size(0), -1) #torch.Size([Batch,time*dmodel])
        out = self.linear(out)
        return out
