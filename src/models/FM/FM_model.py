import numpy as np
import torch
import torch.nn as nn


# factorization을 통해 얻은 feature를 embedding 합니다.
class FeaturesEmbedding(nn.Module):
    """
    Embeds input features using factorization.

    Args:
        field_dims (np.ndarray): An array containing the dimensions of each field
        embed_dim (int): The dimension of the embedding
    """    
    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)


    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


# FM모델 등에서 활용되는 선형 결합 부분을 정의합니다.
class FeaturesLinear(nn.Module):
    """
    Linear Combination

    Args:
        field_dims (np.ndarray): An array containing the dimensions of each field
        output_dim (int): The dimension of the output
    """
    def __init__(self, field_dims: np.ndarray, output_dim: int=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)


    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


# feature 사이의 상호작용을 효율적으로 계산합니다.
class FactorizationMachine(nn.Module):
    """
    Calculate interactions between features

    Args:
        reduce_sum (bool): Determines whether to sum the results of interactions
    """
    def __init__(self, reduce_sum:bool=True):
        super().__init__()
        self.reduce_sum = reduce_sum


    def forward(self, x: torch.Tensor):
        square_of_sum = torch.pow(torch.sum(x, dim=1), 2)
        sum_of_square = torch.sum(torch.pow(x, 2), dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix

# FM 모델을 구현합니다.
class FactorizationMachineModel(nn.Module):
    """
    Implements FM model
    Consists of feature embedding, linear combination, and factorization machine

    Args:
        args: Parameters needed for model initialization
        data: Data that is input to the model
    """
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        self.linear = FeaturesLinear(self.field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)


    def forward(self, x: torch.Tensor):
        x = self.linear(x) + self.fm(self.embedding(x))
        # return torch.sigmoid(x.squeeze(1))
        return x.squeeze(1)
