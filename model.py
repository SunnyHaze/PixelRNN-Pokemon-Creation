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

        # 定义输出层
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # LSTM前向传播
        out, hidden = self.lstm(x)
        
        # 使用全连接层进行预测
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size, device):
        # 初始化隐藏状态
        return (torch.zeros(1, batch_size, self.hidden_size, device = device),
                torch.zeros(1, batch_size, self.hidden_size, device = device))