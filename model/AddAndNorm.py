import torch.nn as nn

class AddAndNorm(nn.Module):
    '''
    Add and Layer Normalization module for transformer models.

    Arguments:
        dmodel: Dimension of the model

    Methods:
        forward(x, residual): Add the input tensor x and the residual tensor, then apply layer normalization
    '''
    def __init__(self, dmodel):
        super(AddAndNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(dmodel)

    def forward(self, x, residual):
        '''
        Add the input tensor x and the residual tensor, then apply layer normalization.

        Arguments:
            x: Input tensor
            residual: Residual tensor to be added to the input tensor

        Returns:
            Tensor after addition and layer normalization
        '''
        return self.layer_norm(x + residual)
