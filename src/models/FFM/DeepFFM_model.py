import numpy as np
import torch
import torch.nn as nn


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
        self.input_dim = sum(field_dims)
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(self.input_dim, embed_dim) for _ in range(self.num_fields)
        ])
        [torch.nn.init.xavier_uniform_(embedding.weight.data) for embedding in self.embeddings]

    def forward(self, x: torch.Tensor):
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]
        # 매번 사전에 정의된 필드만으로 계산이 되는 것을 확인할 수 있음
        ix = [xs[j][:, i] * xs[i][:, j] for i in range(self.num_fields - 1) for j in range(i + 1, self.num_fields)]
        ix = torch.stack(ix, dim=1)
        return ix