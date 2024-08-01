import torch.nn as nn

class FeedForward(nn.Module):
    '''
    Position-wise Feed-Forward Network for transformer models with dropout.

    Arguments:
        dmodel: Dimension of the model
        dropout: Dropout probability

    Methods:
        forward(x): Apply the feed-forward network with dropout on the input tensor x
    '''
    def __init__(self, dmodel, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(dmodel, dmodel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dmodel, dmodel)

    def forward(self, x):
        '''
        Apply the feed-forward network with dropout on the input tensor x.

        Arguments:
            x: Input tensor

        Returns:
            Tensor after applying the feed-forward network and dropout
        '''
        return self.linear2(self.dropout(self.relu(self.linear1(x))))
