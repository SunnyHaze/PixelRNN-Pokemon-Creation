from torch.utils.data import Dataset
import torch
import random

from . import img_reader

class custom_dataset(Dataset):
    def __init__(self, path, seq_len = 80, color_dim = 167):
        # re-manage datasets, transform into one-hot tensors
        pixel_color_data = torch.tensor(img_reader.txt_matrix_reader(path), dtype=torch.long)
        
        id_matrix = torch.eye(color_dim)
        self.seq_len = seq_len
        self.one_hot_data = id_matrix[pixel_color_data] # shape: 792 400 167
        
    def __getitem__(self, idx):
        sample = self.one_hot_data[idx]
        start_p_idx = torch.randint(0, 400 - self.seq_len - 1, (1,)) # minus 1 to give space to label
        stop_p_idx = start_p_idx + self.seq_len
        # print(start_p_idx, stop_p_idx)
        label = sample[stop_p_idx + 1]
        
        if self.seq_len == -1 or self.seq_len < 0:
            return sample
        else:
            return idx, sample[start_p_idx : stop_p_idx], label
        
    def __len__(self):
        return len(self.one_hot_data)
    
if __name__ == "__main__":
    data = custom_dataset("pixel_color.txt")
    import pdb
    pdb.set_trace()