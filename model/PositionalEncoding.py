import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    '''
    Converts the vector embedding of a batch of sequences to their positional encoding vectors.

    Arguments:
            shape : shape of embedding vector => tuple(batch_size, max_len, dmodel)
            device : device to perform the computation on (e.g., 'cpu' or 'cuda')

    Returns:
            positional encoded vector

    '''
    def __init__(self, shape, device):
        super(PositionalEncoding, self).__init__()
        self.max_len = shape[1]
        self.dmodel = shape[2]
        self.device = device

        position = torch.arange(0, self.max_len, device=self.device).float().unsqueeze(1)        
        
        div_term = torch.exp(torch.arange(0, self.dmodel, 2, device=self.device).float() * -(math.log(10000.0) / self.dmodel))

        pos_enc = torch.zeros((1, self.max_len, self.dmodel), device=self.device)
        pos_enc[0, :, 0::2] = torch.sin(position * div_term)
        pos_enc[0, :, 1::2] = torch.cos(position * div_term)

        self.pos_enc = pos_enc

    def forward(self, x):
        x = x + self.pos_enc[:, :x.size(1), :]
        return x
