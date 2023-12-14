import numpy as np
import torch
import torch.nn as nn


# factorization을 통해 얻은 feature를 embedding 합니다.
class FeaturesEmbedding(nn.Module):
    """
    Embeds input features using factorization.

    Args:
        field_dims (np.ndarray): An array containing the dimensions of input fields.
        embedding_dim (int): The dimension of the embedding space.

    Returns:
        torch.Tensor: Embedded features tensor.
    """
    def __init__(self, field_dims: np.ndarray, embedding_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embedding_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)


    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


# NCF 모델은 MLP와 GMF를 합하여 최종 결과를 도출합니다.
# MLP을 구현합니다.
class MultiLayerPerceptron(nn.Module):
    """
    Forward pass of the FeaturesEmbedding module.

    Args:
        x (torch.Tensor): Input tensor representing categorical field indices.

    Returns:
        torch.Tensor: Embedded features tensor.
    """
    def __init__(self, input_dim, embedding_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embedding_dim in embedding_dims:
            layers.append(torch.nn.Linear(input_dim, embedding_dim))
            layers.append(torch.nn.BatchNorm1d(embedding_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embedding_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)


    def forward(self, x):
        return self.mlp(x)


# user와 item의 latent factor를 활용하여 GMF를 구현합니다.
# 그리고 MLP결과와 concat하여 NCF 모델을 구현하고 최종 결과를 도출합니다.
class NeuralCollaborativeFiltering(nn.Module):
    """
    Multi-Layer Perceptron (MLP) implementation.

    Args:
        input_dim (int): Dimension of the input features.
        embedding_dims (list): List of dimensions for each embedding layer in the MLP.
        dropout (float): Dropout probability.
        output_layer (bool): Flag to include an output layer.

    Returns:
        torch.Tensor: Output tensor.
    """
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.user_field_idx = np.array((0, ), dtype=np.int32)
        self.item_field_idx = np.array((1, ), dtype=np.int32)
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        self.embed_output_dim = len(self.field_dims) * args.embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, args.mlp_dims, args.dropout, output_layer=False)
        self.fc = torch.nn.Linear(args.mlp_dims[-1] + args.embed_dim, 1)


    def forward(self, x):
        x = self.embedding(x)
        user_x = x[:, self.user_field_idx].squeeze(1)
        item_x = x[:, self.item_field_idx].squeeze(1)
        gmf = user_x * item_x
        x = self.mlp(x.view(-1, self.embed_output_dim))
        x = torch.cat([gmf, x], dim=1)
        x = self.fc(x).squeeze(1)
        return x
