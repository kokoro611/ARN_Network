import torch
from net.NetModel import Net
from net.NetModel_mine import Net_mine
import time
from torchvision.utils import save_image
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2

x = torch.rand((1, 3, 128, 128)).cuda()
time_list = []
for i in range(1000):
    torch.cuda.synchronize()
    net = Net_mine().cuda()
    start = time.time()
    result = net(x)
    torch.cuda.synchronize()

    end = time.time()
    time_list.append((end-start))
    # print((end-start)*1000)
print(np.mean(time_list)*1000)
