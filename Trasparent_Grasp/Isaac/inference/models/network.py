import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.models.grasp_model import GraspModel, ResidualBlock


class TGCNN(GraspModel):

    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0):
        super(TGCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(channel_size)

        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size * 2)

        self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size * 4)
        self.res1 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res2 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res3 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res4 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res5 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.conv4 = nn.ConvTranspose2d(channel_size * 8, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size * 2)

        self.conv5 = nn.ConvTranspose2d(channel_size * 4, channel_size, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channel_size)

        self.conv6 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=9, stride=1, padding=4)

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=3, padding=1)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        x_32in = F.relu(self.bn1(self.conv1(x_in)))  # b,32,224,224
        x_64in = F.relu(self.bn2(self.conv2(x_32in)))  # b,64,112,112
        x_128in = F.relu(self.bn3(self.conv3(x_64in)))  # b,128,56,56
        x_res1 = self.res1(x_128in)  # b,128,56,56
        x_res2 = self.res2(x_res1)  # b,128,56,56
        x_res3 = self.res3(x_res2)  # b,128,56,56
        x_res4 = self.res4(x_res3)  # b,128,56,56
        x_res5 = self.res5(x_res4)  # b,128,56,56

        x_64out = F.relu(self.bn4(self.conv4(torch.cat([x_res5, x_128in], 1))))  # b,64,112,112
        x_32out = F.relu(self.bn5(self.conv5(torch.cat([x_64out, x_64in], 1))))  # b,32,224,224
        x_out = self.conv6(torch.cat([x_32out, x_32in], 1))  # b,32,224,224

        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x_out))  # b,1,224,224
        else:
            pos_output = self.pos_output(x_out)

        # # 应用二值化阈值
        # threshold = 0.5  # 可以根据需要调整这个阈值
        # output = (pos_output > threshold).float()


        return pos_output
