# from datasets.img_reader import colormap, pixel_color
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.nn import functional as F
from torch import nn as nn
from torch.utils.data import random_split


from utils.data_utils import custom_dataset


class PixelRNN(nn.Module):
    def __init__(self, input_size = 167, hidden_size=512):
        super(PixelRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # 定义输出层
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, hidden):
        # LSTM前向传播
        out, hidden = self.lstm(x, hidden)
        
        # 使用全连接层进行预测
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        # 初始化隐藏状态
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))


my_dataset = custom_dataset("pixel_color.txt")

train_ratio = 0.95
test_ratio = 1 - train_ratio

train_size = int(train_ratio * len(my_dataset))
test_size = len(my_dataset) - train_size

train_dataset, test_dataset = random_split(my_dataset, [train_size, test_size])


print(len(train_dataset))
print(len(test_dataset))




# # re-manage datasets, transform into one-hot tensors
# pixel_color_data = torch.tensor(pixel_color, dtype=torch.long)
# print(pixel_color_data.shape)
# one_hot_matrix = torch.eye(167)

# one_hot_imgs = one_hot_matrix[pixel_color_data]
# print(one_hot_imgs.shape)

# print(one_hot_imgs[0])


# pixel_imgs = torch.argmax(one_hot_imgs, dim=-1)
# print(pixel_imgs.shape)
# print(torch.sum(pixel_imgs - pixel_color_data)) 



# batchsize = 4


# model = PixelRNN()
# print(model)

# hidden = model.init_hidden(1)
# train_X = one_hot_imgs[0:1]
# print(train_X.shape)
# output, hidden = model(train_X, hidden)
# print(output.shape)
# print(output)