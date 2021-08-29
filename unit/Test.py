import torch
from net.NetModel import Net
from unit.DataTest import MyTestDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# 测试图像的路径
input_path = 'rainy_image_dataset/testing/rainy_image/'

net = Net().cuda()
net.load_state_dict(torch.load('./model.pth')) # 加载训练好的模型参数
net.eval()

cnt = 0

dataloader = DataLoader(MyTestDataset(input_path))
for input in dataloader:
    cnt += 1
    input = input.cuda()

    print('finished:{:.2f}%'.format(cnt*100/1400))

    with torch.no_grad():
        output_image = net(input) # 输出的是张量
        save_image(output_image, 'result/'+str(cnt).zfill(4)+ '_result' +'.jpg') # 直接保存张量图片，自动转换
