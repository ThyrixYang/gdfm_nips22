import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, input_size, output_size, hidden_sizes=[128, 128, 128], bn=True, dropout=0):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.bn = bn
        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_size))
            else:
                self.layers.append(
                    nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            self.layers.append(nn.ReLU())
            if bn:
                self.layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x, _=None):
        if len(x.shape) > 1:
            x = x.view(-1, self.input_size)
        x = x.float()
        for layer in self.layers[:-1]:
            x = layer(x)
        out = self.layers[-1](x)
        return out

num_bin_size = (64, 16, 128, 64, 128, 64, 512, 512)
cate_bin_size = (512, 128, 256, 256, 64, 256, 256, 16, 256)

class CriteoMLP(nn.Module):

    def __init__(self, 
                 output_size, 
                 hidden_sizes=[256, 256, 128], 
                 embedding_size=16,
                 bn=True, 
                 dropout=0):
        super(CriteoMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.input_size = 17
        self.embedding_num = (*cate_bin_size, *num_bin_size)
        self.embedding_size = embedding_size
        self.embeddings = nn.ModuleList()
        for i in range(len(self.embedding_num)):
            self.embeddings.append(nn.Embedding(self.embedding_num[i], self.embedding_size))
        self.bn = bn
        net_input_size = len(self.embedding_num) * self.embedding_size
        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                self.layers.append(nn.Linear(net_input_size, hidden_size))
            else:
                self.layers.append(
                    nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            self.layers.append(nn.ReLU())
            if bn:
                self.layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x, _=None, return_emb=False):
        if len(x.shape) > 1:
            x = x.view(-1, self.input_size)
        mlp_input = []
        for i, embedding in enumerate(self.embeddings):
            emb_i = embedding(x[:, i])
            mlp_input.append(emb_i)
        x_emb = torch.cat(mlp_input, dim=1)
        x = x_emb
        for layer in self.layers[:-1]:
            x = layer(x)
        out = self.layers[-1](x)
        if return_emb:
            return out, x_emb
        else:
            return out
        
class CriteoDXY(nn.Module):
    
    def __init__(self, 
                 y_size, 
                 d_size,
                 hidden_size=128):
        super(CriteoDXY, self).__init__()
        self.y_size = y_size
        self.d_size = d_size
        self.hidden_size = hidden_size
        
        self.x_encoder = CriteoMLP(self.hidden_size, [self.hidden_size, self.hidden_size, self.hidden_size], bn=True)
        self.d_net = MLP(input_size=self.y_size + self.hidden_size, 
                         output_size=self.d_size,
                         hidden_sizes=[self.hidden_size, self.hidden_size, self.hidden_size], 
                         bn=True)
        
    def forward(self, x, y):
        x_emb = self.x_encoder(x, return_emb=False)
        xy_input = torch.cat([x_emb, y], dim=1)
        d_pred = self.d_net(xy_input)
        return d_pred
    
alimama_bin_size = (1000, 10000, 1000)
class TaobaoMLP(nn.Module):

    def __init__(self, 
                 output_size, 
                 hidden_sizes=[256, 256, 128], 
                 embedding_size=32,
                 bn=True, 
                 dropout=0):
        super(TaobaoMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.input_size = 18
        self.embedding_num = alimama_bin_size
        self.embedding_size = embedding_size
        self.embeddings = nn.ModuleList()
        for i in range(len(self.embedding_num)):
            self.embeddings.append(nn.Embedding(self.embedding_num[i], self.embedding_size))
        self.bn = bn
        net_input_size = len(self.embedding_num) * self.embedding_size + 2 * 5 * self.embedding_size + 4 * 5
        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                self.layers.append(nn.Linear(net_input_size, hidden_size))
            else:
                self.layers.append(
                    nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            self.layers.append(nn.ReLU())
            if bn:
                self.layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x, _=None, return_emb=False):
        batch_size = x.shape[0]
        if len(x.shape) > 1:
            x = x.view(-1, self.input_size)
        mlp_input = []
        for i, embedding in enumerate(self.embeddings):
            emb_i = embedding(x[:, i])
            mlp_input.append(emb_i)
        x_hist = x[:, 3:].view((-1, 5, 3))
        hist_item_emb = self.embeddings[1](x_hist[:, :, 0]).view((batch_size, -1))
        hist_item_cate_emb = self.embeddings[2](x_hist[:, :, 1]).view((batch_size, -1))
        hist_type_emb = F.one_hot(x_hist[:, :, 2], num_classes=4).view((batch_size, -1))
        mlp_input.append(hist_item_emb)
        mlp_input.append(hist_item_cate_emb)
        mlp_input.append(hist_type_emb)
        
        x_emb = torch.cat(mlp_input, dim=1)
        x = x_emb
        for layer in self.layers[:-1]:
            x = layer(x)
        out = self.layers[-1](x)
        if return_emb:
            return out, x_emb
        else:
            return out
        
class TaobaoDXY(nn.Module):
    
    def __init__(self, 
                 y_size, 
                 d_size,
                 hidden_size=256):
        super(TaobaoDXY, self).__init__()
        self.y_size = y_size
        self.d_size = d_size
        self.hidden_size = hidden_size
        
        self.x_encoder = TaobaoMLP(self.hidden_size, [self.hidden_size, self.hidden_size, self.hidden_size], bn=True)
        self.d_net = MLP(input_size=self.y_size + self.hidden_size, 
                         output_size=self.d_size,
                         hidden_sizes=[self.hidden_size, self.hidden_size, self.hidden_size], 
                         bn=True)
        
    def forward(self, x, y):
        x_emb = self.x_encoder(x, return_emb=False)
        xy_input = torch.cat([x_emb, y], dim=1)
        d_pred = self.d_net(xy_input)
        return d_pred