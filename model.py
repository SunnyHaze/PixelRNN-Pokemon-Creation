import torch
from torch.nn import functional as F
from torch import nn as nn

class PixelRNN(nn.Module):
    def __init__(self, input_size = 167, hidden_size=512):
        super(PixelRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # self.norm = nn.BatchNorm1d(512)
        # 定义输出层
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, hidden = None):
        # LSTM前向传播
        if hidden != None:
            out, hidden = self.lstm(x, hidden)
        else:
            out, hidden = self.lstm(x)
        
        # print(out.shape)  # 1, 40, 512
        # out = self.norm(out.permute(0, 2, 1)).permute(0, 2, 1)
        # 使用全连接层进行预测
        out = self.fc(out)
        
        # out = torch.softmax(out, dim=1)
        return out, hidden

    def init_hidden(self, batch_size, device):
        # 初始化隐藏状态
        return (torch.zeros(1, batch_size, self.hidden_size, device = device),
                torch.zeros(1, batch_size, self.hidden_size, device = device))