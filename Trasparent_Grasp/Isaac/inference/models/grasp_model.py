import torch.nn as nn
import torch.nn.functional as F



class GraspModel(nn.Module):
    def __init__(self):
        super(GraspModel, self).__init__()

    def forward(self, x_in):
        raise NotImplementedError()

    def compute_loss(self, xc, yc):
        y_pos, y_radius = yc
        pos_pred, radius_pred = self(xc)
        p_loss = F.smooth_l1_loss(pos_pred, y_pos)
        radius_loss = F.smooth_l1_loss(radius_pred, y_radius)

        return {
            'loss': p_loss + radius_loss,
            'losses': {
                'p_loss': p_loss,
                'radius_loss': radius_loss
            },
            'pred': {
                'pos': pos_pred,
                'radius': radius_pred
            }
        }

    def predict(self, xc):
        pos_pred, width_pred = self(xc)
        return {
            'pos': pos_pred,
            'width': width_pred
        }


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in
