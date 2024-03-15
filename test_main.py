# from datasets.img_reader import colormap, pixel_color
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.nn import functional as F
from torch import nn as nn
from torch import optim as optim 
from torch.utils.data import random_split, DataLoader
from pathlib import Path

from utils.data_utils import custom_dataset
from model import PixelRNN

def get_args_parser():
    parser = argparse.ArgumentParser('Pokemon Generator by LSTM Training', add_help=True)
    parser.add_argument("--batch_size", default=1, type=int,
                        help="Batchsize per GPU")
    parser.add_argument("--output_dir", default="output_dir_1", type= str,
                        help = 'output dir for ckpt and logs')
    parser.add_argument("--epoch", default=100, type=int,
                        help = 'Number of epochs')
    parser.add_argument("--lr", default="1e-4", type=float,
                        help = 'Learning rate')
    parser.add_argument("--device", default="cuda", type=str,
                        help="Device: cuda or GPU")
    return parser

def generate_image(model, initial_pixels, max_pixels=400):
    # 初始化生成图像
    generated_image = initial_pixels
    
    # 将生成的像素逐步添加到图像中，直到达到最大像素数量
    for _ in range(max_pixels - len(initial_pixels)):
        # 将生成图像转换为模型的输入格式
        input_tensor = torch.tensor(generated_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # 使用模型进行预测
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        # 从输出中获取下一个像素的概率分布
        next_pixel_probs = output[:, -1, :]
        
        # 根据概率分布采样下一个像素值
        next_pixel = torch.multinomial(next_pixel_probs, 1).item()
        
        # 将下一个像素添加到生成的图像中
        generated_image.append(next_pixel)
    
    return generated_image
    

def main(args):
    log_writer = SummaryWriter(log_dir=args.output_dir)
    
    my_dataset = custom_dataset("pixel_color.txt")

    train_ratio = 0.995
    test_ratio = 1 - train_ratio

    train_size = int(train_ratio * len(my_dataset))
    test_size = len(my_dataset) - train_size

    train_dataset, test_dataset = random_split(my_dataset, [train_size, test_size])

    print(len(train_dataset))
    print(len(test_dataset))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size = args.batch_size, 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    device = args.device
    model = PixelRNN()
    
    model = model.to(device)
    print(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    
    # hidden state for LSTM
    # hidden = model.init_hidden(args.batch_size, device = device)
    # hidden = hidden.to(device)
    
    for epoch in range(args.epoch):
        running_loss = 0.0
        
        # train
        model.train()
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            pred, hidden = model(data)
            
            pred_pixel = pred[:, -1, :] # Batch, seq_len, dim
            loss = criterion(
                pred_pixel.view(-1, pred_pixel.size(-1) ),  # B, Dim
                label.view(-1, label.size(-1) )             # B, Dim
                )
            # print(loss)

            loss.backward()
            optimizer.step()
            running_loss += loss
        log_writer.add_scalar("train/epoch_loss", running_loss, epoch)
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
        log_writer.flush()

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
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