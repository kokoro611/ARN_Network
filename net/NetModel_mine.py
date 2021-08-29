import torch
import torch.nn as nn
import torch.nn.functional as F
#from tensorboardX import SummaryWriter
#from torchviz import make_dot
from net.aspp import ASPP

class Net_mine(nn.Module):
    def __init__(self):
        super(Net_mine, self).__init__()
        self.conv0_0 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1), # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1
            nn.ReLU()
        )
        self.res_conv0_1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv0_2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv0_3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv0_4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv0_5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv0 = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )
        #########################################################
        self.conv1_0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1), # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1
            nn.ReLU()
        )
        self.res_conv1_1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1_2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1_3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1_4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1_5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )

        self.aspp=ASPP()

    def forward(self, input):
        x = input

        x = torch.cat((input, x), 1)


        x = self.aspp(x)



        x = self.conv0_0(x)
        x = F.relu(self.res_conv0_1(x) + x)
        x = F.relu(self.res_conv0_2(x) + x)
        x = F.relu(self.res_conv0_3(x) + x)
        x = F.relu(self.res_conv0_4(x) + x)
        x = F.relu(self.res_conv0_5(x) + x)
        x = self.conv0(x)
        x = x + input

        x_res = x
        x = torch.cat((input, x), 1)
        x = self.conv1_0(x)
        x = F.relu(self.res_conv1_1(x) + x)
        x = F.relu(self.res_conv1_2(x) + x)
        x = F.relu(self.res_conv1_3(x) + x)
        x = F.relu(self.res_conv1_4(x) + x)
        x = F.relu(self.res_conv1_5(x) + x)
        x = self.conv1(x)
        x = x + x_res

        return x

if __name__ == "__main__":
    x = torch.rand((1, 3, 224, 224))
    net = Net_mine()
    out = net(x)
    print(net)

    '''
    g = make_dot(out)
    g.render('espnet_model', view=False)

    with SummaryWriter(comment='resnet') as w:
        w.add_graph(net, x)
    '''