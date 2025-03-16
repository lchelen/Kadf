import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable

class encoder(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.encoder1 = nn.Linear(input_size, hidden_size, bias=bias)
        self.encoder2 = nn.Linear(input_size, hidden_size, bias=bias)
        self.decoder = nn.Linear(hidden_size, input_size, bias=bias)

    def forward(self, x):
        encode1 = self.encoder1(x)
        encode2 = self.encoder2(x)
        decode = self.decoder(encode1 + encode2)
        return encode1, encode2, decode
    
    
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            nn.init.uniform_(w, -std, std)

    def forward(self, x, hidden):
        hx, cx = hidden
        x = x.view(-1, x.size(1))
        gates = self.x2h(x) + self.h2h(hx)
        gates = gates.squeeze()
        
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, F.tanh(cy))
        return (hy, cy)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = LSTMCell(input_dim, hidden_dim, layer_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        if torch.cuda.is_available():
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        outs = []
        cn_outs = torch.Tensor()
        cn = c0[0, :, :]
        hn = h0[0, :, :]

        for seq in range(x.size(1)):
            hn, cn = self.lstm(x[:, seq, :], (hn, cn))
            outs.append(hn)
            cn_outs = torch.cat((cn_outs, cn.unsqueeze(1)), 1)
        out = outs[-1].squeeze()
        out = self.fc2(out)
        return out, cn_outs
    

class TMUModel(nn.Module):
    def __init__(self, hidden_dim, layer_dim, output_dim, bias=True):
        super(TMUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm1 = LSTMCell(hidden_dim, hidden_dim, layer_dim)
        self.lstm2 = LSTMCell(hidden_dim, hidden_dim, layer_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x1, x2, memory):

        h01 = Variable(torch.zeros(self.layer_dim, x1.size(0), self.hidden_dim))
        c01 = Variable(torch.zeros(self.layer_dim, x1.size(0), self.hidden_dim))
        h02 = Variable(torch.zeros(self.layer_dim, x2.size(0), self.hidden_dim))
        c02 = Variable(torch.zeros(self.layer_dim, x2.size(0), self.hidden_dim))
        
        outs1 = []
        outs2 = []
        cn1 = c01[0, :, :]
        hn1 = h01[0, :, :]
        cn2 = c02[0, :, :]
        hn2 = h02[0, :, :]
        loss = 0
        for seq in range(x1.size(1)):
            hn1, cn1 = self.lstm1(x1[:, seq, :], (hn1, cn1))
            hn2, cn2 = self.lstm2(x2[:, seq, :], (hn2, cn2))
            outs1.append(hn1)
            outs2.append(hn2)
            loss = loss + torch.mean((cn2 - memory[:, seq, :]).pow(2))
        out1 = outs1[-1].squeeze()
        out2 = outs2[-1].squeeze()
        out = self.fc(out1 + out2)
        return out, loss
