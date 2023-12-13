'''
CNN-FM
'''
import numpy as np
import torch
from torch import nn

class FactorizationMachine(nn.Module):
    
    def __init__(self, input_dim, latent_dim):

        super(FactorizationMachine, self).__init__()
        self.v = nn.Parameter(torch.rand(input_dim, latent_dim), requires_grad=True)
        self.linear = nn.Linear(input_dim, 1)
        self.init_params()

    def init_params(self):
        for layer in self.children():
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):

        # linear term
        linear = self.linear(x)

        # fm term
        square_of_sum = torch.mm(x, self.v)**2
        sum_of_square = torch.mm(x**2, self.v**2)

        x = linear + 0.5 * torch.sum(square_of_sum + sum_of_square, dim=1, keepdim=True)
        return x

class CNN_branch(nn.Module):
    
    def __init__(self):

        super(CNN_branch, self).__init__()

        kernel_size = 3
        stride = 2
        padding = 1

        self.cnn1 = nn.Conv2d(3, 6, kernel_size, stride, padding)
        self.maxpool1 = nn.MaxPool2d(kernel_size, stride)

        self.cnn2 = nn.Conv2d(6, 12, kernel_size, stride, padding)
        self.maxpool2 = nn.MaxPool2d(kernel_size, stride)

        self.relu = nn.ReLU()

    def init_params(self):
        for layer in self.children():
            if type(layer) == nn.Conv2D:
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):

        x = self.cnn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.cnn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        x = x.view(-1, 12)

        return x


class CNN_FM(nn.Module):
    
    def __init__(self, args, data):

        super(CNN_FM, self).__init__()

        field_dims = np.array(
            [len(data['idx2user']), len(data['idx2isbn'])], dtype=np.uint32)
        self.embedding = torch.nn.Embedding(sum(field_dims), args.cnn_embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)

        self.cnn = CNN_branch()
        self.fm = FactorizationMachine(
                args.cnn_embed_dim*2+args.cnn_latent_dim, args.cnn_latent_dim)

        torch.nn.init.xavier_uniform_(self.embedding.weight.data)


    def forward(self, x):

        user_isbn_vector, img_vector = x[0], x[1]

        # get user isbn feature
        user_isbn_vector = user_isbn_vector + \
            user_isbn_vector.new_tensor(self.offsets).unsqueeze(0)
        user_isbn_feature = self.embedding(user_isbn_vector)

        # get img feature
        img_feature = self.cnn(img_vector)

        # concat embeddings
        x = torch.cat((user_isbn_feature.view(-1,2*64), img_feature), 1)

        x = self.fm(x)

        return x.squeeze(1)
