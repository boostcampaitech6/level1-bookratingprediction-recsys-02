import numpy as np
import torch
from torch import nn as nn


class FM_component(nn.Module):
    ''' 
    FM component
    :param input_dim: 입력 벡터의 차원
    '''

    def __init__(self, input_dim):
        super(FM_component, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self._init_params()

    def _init_params(self):
        for child in self.children():
            if isinstance(child, nn.Linear):
                nn.init.uniform_(child.weight)
                nn.init.zeros_(child.bias)


    def forward(self, sparse_x, embed_x):
        '''
        :param sparse_x:  multihot vector (batch_size, input_dim)
        :param embed_x: embedded x vectors (batch_size, num_fields, factor_dim)
        :return: output: FM 레이어의 출력값
                       Float Tensor이며 사이즈는 "(batch_size, 1)"
        '''
        # w0x linear
        linear_output = self.linear(sparse_x)

        # 교호작용
        square_of_sum = torch.sum(embed_x, dim=1)**2
        sum_of_square = torch.sum(embed_x**2, dim=1)
        fm_output = .5 * torch.sum((square_of_sum - sum_of_square), dim=1)

        # linear + fm_term
        output = linear_output.squeeze(1) + fm_output

        return output


def activation_layer(fname):
    
    if fname == 'relu':
        activation_layer = nn.ReLU()
    elif fname == 'tanh':
        activation_layer = nn.Tanh()
    else:
        raise NameError
    
    return activation_layer


class DNN_component(nn.Module):
    '''
    여러 개의 fully connected layer를 거쳐 출력하는 DNN 레이어
    :param input_dim: 입력 차원 (field_nums * factor)
    :param mlp_dims: 각 mlp layer의 차원들이 담긴 list
    :param dropout_rate: dropout 비율
    :param activation: activation 함수 이름 (str)
    :param use_bn: batch normalization 사용 여부
    '''
    def __init__(self, input_dim, mlp_dims, dropout_rate=0, 
            activation_name='relu', use_bn=False):

        super().__init__()
        
        # stack dnn layers
        self.layers = nn.Sequential()
        
        prev_dim = input_dim 
        for dim in mlp_dims:
            self.layers.append(nn.Linear(prev_dim, dim))
            self.layers.append(activation_layer(activation_name))
            if use_bn:
                self.layers.append(nn.BatchNorm1d(dim))
            self.layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = dim 

        # last layer
        self.last_layer = nn.Linear(mlp_dims[-1], 1, bias=False)

        # init params
        self._init_params()
        
    def _init_params(self):
        for child in self.children():
            if isinstance(child, nn.Linear):
                nn.init.xavier_uniform_(child.weight)
                if child.bias:
                    nn.init.zeros_(child.bias)

    def forward(self, x):
        '''
        :param x: 임베딩을 펼친 텐서. 사이즈는 "(batch_size, num_fields*factor dim)"
        :return: y_dnn: DNN component의 출력값
                    Float Tensor이며 사이즈는 "(batch_size, 1)"
        '''
        x = self.layers(x)
        output = self.last_layer(x)
        return output


class DeepFMModel(nn.Module):
    '''
    DeepFM 모델
    :param args: 하이퍼파라미터 등 인자를 담은 객체. 속성으로 접근 
    :param data: 각종 데이터 정보를 담은 dict
    '''
    def __init__(self, args, data):
        super(DeepFMModel, self).__init__()

        factor_dim = args.embed_dim
        field_dims = data['field_dims']
        
        # 각종 변수들을 설정
        self.input_dim = sum(field_dims) # 입력값의 차원 = 모든 field의 크기를 더한 값
        self.num_fields = len(field_dims) # field의 개수
        self.encoding_dims = np.concatenate([[0], np.cumsum(field_dims)[:-1]]) # 각 field의 시작 위치
        
        # 각 field에 대한 임베딩 레이어를 담은 리스트
        self.embedding = nn.ModuleList([
            nn.Embedding(self.input_dim, factor_dim) for feature_size in field_dims
        ])
        
        # FM component
        self.fm = FM_component(input_dim=self.input_dim)

        # DNN component 
        self.dnn = DNN_component(
            input_dim=(self.num_fields * factor_dim), 
            mlp_dims=args.mlp_dims, activation_name=args.activation_fn, 
            dropout_rate=args.dropout, use_bn=args.use_bn)
        
        self._init_params()
        
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
        
    def forward(self, x):
        '''
        :param x: n차원 정수형(Long) 입력 텐서. 사이즈는 "(batch_size, num_fields)"
            sparse_x : x를 one-hot-encoding한 sparse tensor
                      Float Tensor이며 사이즈는 "(batch_size, input_dim)"
            dense_x  : x를 field별로 embedding한 dense tensor
                       "num_fields"개의 Float Tensor로 이루어진 리스트로, 
                       각 Float Tensor의 사이즈는 "(batch_size, factor_dim)"
        :return: y: 모델의 출력값
                    Float Tensor이며 사이즈는 "(batch_size, 1)"
        '''
        # sparse_x 만들기
        # [기본과제3]의 FieldFM에서 사용한 x_multihot과 동일한 코드
        sparse_x = x + x.new_tensor(self.encoding_dims).unsqueeze(0)
        sparse_x = torch.zeros(x.size(0), self.input_dim, device=x.device).scatter_(1, x, 1.)

        # dense_x 만들기
        # [기본과제3]의 FFMLayer에서 사용한 xv와 동일한 코드
        dense_x = [self.embedding[f](x[...,f]) for f in range(self.num_fields)] 
        
        # FM 레이어를 거쳐 y_fm을 구함
        y_fm = self.fm(sparse_x, torch.stack(dense_x, dim=1))

        # DNN 레이어를 거쳐 y_dnn을 구함
        y_dnn = self.dnn(torch.cat(dense_x, dim=1))
        
        # y = y_fm + y_dnn
        y = y_fm + y_dnn.squeeze(1)

        return y

class DeepFMModel(nn.Module):
    '''
    DeepFM 모델
    :param args: 하이퍼파라미터 등 인자를 담은 객체. 속성으로 접근 
    :param data: 각종 데이터 정보를 담은 dict
    '''
    def __init__(self, args, data):
        super(DeepFMModel, self).__init__()

        factor_dim = args.embed_dim
        field_dims = data['field_dims']
        
        # 각종 변수들을 설정
        self.text = args.merge_summary
        if self.text:
            self.text_dim = field_dims[-1]
            field_dims = field_dims[:-1]

            # text layer
            self.text_layer = nn.Sequential()
            self.text_layer.append(nn.Linear(self.text_dim, 256))
            self.text_layer.append(nn.ReLU())
            self.text_layer.append(nn.Linear(256, 64))
            self.text_layer.append(nn.ReLU())

        self.input_dim = sum(field_dims) # 입력값의 차원 = 모든 field의 크기를 더한 값
        self.num_fields = len(field_dims) # field의 개수
        self.offsets = np.concatenate([[0], np.cumsum(field_dims)[:-1]]) # 각 field의 시작 위치
        
        # 각 field에 대한 임베딩 레이어를 담은 리스트
        self.embedding = nn.ModuleList([
            nn.Embedding(feature_size, factor_dim) for feature_size in field_dims
        ])

        
        # FM component
        self.fm = FM_component(input_dim=self.input_dim)

        # DNN component 
        dnn_input = self.num_fields * factor_dim
        if self.text: dnn_input += 64#self.text_dim
        self.dnn = DNN_component(
            input_dim=dnn_input,
            mlp_dims=args.mlp_dims, activation_name=args.activation_fn, 
            dropout_rate=args.dropout, use_bn=args.use_bn)
        
        self._init_params()
        
    def _init_params(self):
        for child in self.children():
            if isinstance(child, nn.Embedding):
                nn.init.xavier_uniform_(child.weight)
            elif isinstance(child, nn.Linear):
                nn.init.xavier_uniform_(child.weight)
                if child.bias:
                    nn.init.zeros_(child.bias)

    def forward(self, x):
        '''
        :param x: n차원 정수형(Long) 입력 텐서. 사이즈는 "(batch_size, num_fields)"
            sparse_x : x를 one-hot-encoding한 sparse tensor
                      Float Tensor이며 사이즈는 "(batch_size, input_dim)"
            dense_x  : x를 field별로 embedding한 dense tensor
                       "num_fields"개의 Float Tensor로 이루어진 리스트로, 
                       각 Float Tensor의 사이즈는 "(batch_size, factor_dim)"
        :return: y: 모델의 출력값
                    Float Tensor이며 사이즈는 "(batch_size, 1)"
        '''

        if self.text:
            x, text_x = x
            text_vector = self.text_layer(text_x.squeeze(-1))

        # dense_x 만들기
        dense_x = [self.embedding[f](x[...,f]) for f in range(self.num_fields)] 

        # sparse_x 만들기
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        sparse_x = torch.zeros(x.size(0), self.input_dim, device=x.device).scatter_(1, x, 1.)
        
        # FM 레이어를 거쳐 y_fm을 구함
        y_fm = self.fm(sparse_x, torch.stack(dense_x, dim=1))

        # DNN 레이어를 거쳐 y_dnn을 구함
        embed_x = torch.cat(dense_x, dim=1)

        # text_x 차원 줄이기
        if self.text:
            y_dnn = self.dnn(torch.cat((embed_x, text_vector), dim=1))
        else:
            y_dnn = self.dnn(embed_x)
        
        # y = y_fm + y_dnn
        output = y_fm + y_dnn.squeeze(1)

        return output
