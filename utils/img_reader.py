import os
from matplotlib import pyplot as plt
import numpy as np

def txt_matrix_reader(path:str):
    # with open(path, "r") as f:
    #     array = [ [int(i) for i in line.split()] for line in f]
    # return array
    array = np.loadtxt(path, dtype=np.int32)
    return array

colormap = txt_matrix_reader("colormap.txt")
pixel_color = txt_matrix_reader("pixel_color.txt")


def mapping_img(img, shape = (20, 20, 3) ):
    return np.reshape( colormap[img], shape)

if __name__ == "__main__":
    # colormap = txt_matrix_reader("colormap.txt")
    # pixel_color = txt_matrix_reader("pixel_color.txt")
    # print(colormap, pixel_color)
    
    # print(pixel_color.shape)
    # print(colormap.shape)
    
    sample_img = pixel_color[2]
    
    img = mapping_img(sample_img)
    # img = colormap[sample_img]
    # img = np.reshape(img, (20,20,3))
    plt.imshow(img)
    plt.show()
    
    print(img.shape)