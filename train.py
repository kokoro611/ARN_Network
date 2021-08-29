import torch
import torch.optim as optim
from net.NetModel import Net
from net.NetModel_mine import Net_mine
import torch.nn as nn
import os
from unit.DataTrain import MyTrainDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np

## matplotlib显示图片中显示汉字
# plt.rcParams['font.sans-serif'] = ['SimSun']
# plt.rcParams['axes.jiusunicode_minus'] = False
model_path=''
model_path = 'workdirs/model_end3.pth'
# 训练图像的路径
input_path = 'data/train/rain/'
label_path = 'data/train/groundtruth/'

net = Net_mine().cuda()

learning_rate = 7e-3
batch_size = 64# 分批训练数据，每批数据量
epoch = 200 # 训练次数
Loss_list = [] # 简单的显示损失曲线列表，反注释后训练完显示曲线

optimizer = optim.Adam(net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000,gamma=0.9)

loss_f = nn.MSELoss()
loss_l1 = nn.L1Loss()

net.train()

if os.path.exists(model_path):# 判断模型有没有提前训练过
    print("继续训练！")
    net.load_state_dict(torch.load(model_path))# 加载训练过的模型
else:
    print("从头训练！")

for i in range(epoch):
    print('epoch={}, lr={:.6f}'.format(i,scheduler.get_lr()[0]))
    dataset_train = MyTrainDataset(input_path, label_path)
    trainloader = DataLoader(dataset_train, batch_size=batch_size, num_workers=32, shuffle=True)
    pbar = tqdm(total=(len(trainloader)))
    for j, (x, y) in enumerate(trainloader):# 加载训练数据
        input = Variable(x).cuda()
        label = Variable(y).cuda()

        net.zero_grad()
        optimizer.zero_grad()

        output = net(input)
        # if epoch<50:
        loss = loss_f(output, label)



        optimizer.zero_grad()
        loss.backward() # 反向传播
        optimizer.step()  
        scheduler.step()
        pbar.update(1)
        # print("已完成第{}次训练的{:.3f}%，目前损失值为{:.6f}。".format(i+1, ((j+1)/252)*32, loss))
        # print(float(loss))
        Loss_list.append(float(loss))

    pbar.clear()
    pbar.close()
    print('save model, loss={:.8f}'.format(np.mean(Loss_list)))
    torch.save(net.state_dict(), 'workdirs/model_end.pth') # 保存训练模型


# plt.figure(dpi=500)
# x = range(0, 2520*2)
# y = Loss_list
# plt.plot(x, y, 'r-')
# plt.ylabel('当前损失/1')
# plt.xlabel('批训练次数/次数')
# plt.savefig('F://loss.jpg')
# plt.show()