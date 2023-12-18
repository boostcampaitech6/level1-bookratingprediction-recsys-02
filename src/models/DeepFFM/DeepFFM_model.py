import numpy as np
import torch
import torch.nn as nn
from typing import List

# FM모델 등에서 활용되는 선형 결합 부분을 정의합니다.
class FeaturesLinear(nn.Module):
    def __init__(self, field_dims: np.ndarray, output_dim: int=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, x: torch.Tensor):
        return torch.sum(self.fc(x), dim=1) + self.bias

# FFM모델을 구현합니다.
# feature간의 상호작용을 파악하기 위해서 잠재백터를 두는 과정을 보여줍니다.
# FFM은 FM과 다르게 필드별로 여러개의 잠재백터를 가지므로 필드 개수만큼의 embedding parameter를 선언합니다.
class FieldAwareFactorizationMachine(nn.Module):
    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.num_fields = len(field_dims)


    def forward(self, xs: List[torch.Tensor]):
        # 매번 사전에 정의된 필드만으로 계산이 되는 것을 확인할 수 있음
        ix = [xs[j][:, i] * xs[i][:, j] for i in range(self.num_fields - 1) for j in range(i + 1, self.num_fields)]
        ix = torch.stack(ix, dim=1)
        return ix

# MLP을 구현합니다.
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)


    def forward(self, x):
        return self.mlp(x)

# 최종적인 DeepFFM모델입니다.
# 각 필드별로 곱해져 계산된 embedding 결과를 합하고, 마지막으로 embedding 결과를 합하여 마무리합니다.
class DeepFFM(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.num_fields = len(self.field_dims)
        self.input_dim = sum(self.field_dims)
        self.linear = FeaturesLinear(self.field_dims)
        self.ffm = FieldAwareFactorizationMachine(self.field_dims, args.embed_dim)
        self.offsets = np.array((0, *np.cumsum(self.field_dims)[:-1]), dtype=np.int32)
        # 각 field에 대한 임베딩 레이어를 담은 리스트
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(self.input_dim, args.embed_dim) for _ in range(self.num_fields)
        ])
        [torch.nn.init.xavier_uniform_(embedding.weight.data) for embedding in self.embeddings]

        self.embed_output_dim = len(self.field_dims) * args.embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, args.mlp_dims, args.dropout)

    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets, dtype= torch.int32).unsqueeze(0)
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]
        avg_xs = torch.stack(xs, dim=0).sum(dim=0)
        mlp_xs = torch.cat([avg_xs[:, i, :] for i in range(avg_xs.size(1))], dim=1)
        ffm_term = torch.sum(torch.sum(self.ffm(xs), dim=1), dim=1, keepdim=True)
        
        x = self.linear(x) + ffm_term + self.mlp(mlp_xs)
        # return torch.sigmoid(x.squeeze(1))
        return x.squeeze(1)