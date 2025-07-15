import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from einops import rearrange
  
class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, first_dilation=None, dilation=1):
        super(ResBlock, self).__init__()

        self.same_shape = (in_channels == out_channels and stride == 1)


        if first_dilation == None: first_dilation = dilation

        self.bn_branch2a = nn.BatchNorm2d(in_channels)

        self.conv_branch2a = nn.Conv2d(in_channels, mid_channels, 3, stride,
                                       padding=first_dilation, dilation=first_dilation, bias=False)

        self.bn_branch2b1 = nn.BatchNorm2d(mid_channels)

        self.conv_branch2b1 = nn.Conv2d(mid_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)        

    def forward(self, x, get_x_bn_relu=False):

        branch2 = self.bn_branch2a(x)
        branch2 = F.relu(branch2)

        x_bn_relu = branch2

        if not self.same_shape:
            branch1 = self.conv_branch1(branch2)
        else:
            branch1 = x

        branch2 = self.conv_branch2a(branch2)
        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.conv_branch2b1(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)

class ResBlock_bot(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, dropout=0.):
        super(ResBlock_bot, self).__init__()

        self.same_shape = (in_channels == out_channels and stride == 1)

        self.bn_branch2a = nn.BatchNorm2d(in_channels)
        self.conv_branch2a = nn.Conv2d(in_channels, out_channels//4, 1, stride, bias=False)

        self.bn_branch2b1 = nn.BatchNorm2d(out_channels//4)
        self.dropout_2b1 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b1 = nn.Conv2d(out_channels//4, out_channels//2, 3, padding=dilation, dilation=dilation, bias=False)

        self.bn_branch2b2 = nn.BatchNorm2d(out_channels//2)
        self.dropout_2b2 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b2 = nn.Conv2d(out_channels//2, out_channels, 1, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False):

        branch2 = self.bn_branch2a(x)
        branch2 = F.relu(branch2)
        x_bn_relu = branch2

        branch1 = self.conv_branch1(branch2)

        branch2 = self.conv_branch2a(branch2)

        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b1(branch2)
        branch2 = self.conv_branch2b1(branch2)

        branch2 = self.bn_branch2b2(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b2(branch2)
        branch2 = self.conv_branch2b2(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)


class IBM(nn.Module):
    def __init__(self, in_channels, reduction=16, embedding_dim=8):
        super(IBM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.edge = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False)
        self.edge_act = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.spatial_act = nn.Sigmoid()
        self.fuse = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, bias=False)
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.boundary_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1)
        )
        self.embedding_branch = nn.Conv2d(in_channels, embedding_dim, 1)

    def forward(self, x):
        f1 = self.conv1(x)
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        multi_scale = torch.cat([f1, f3, f5], dim=1)
        multi_scale = self.fuse(multi_scale)
        edge_map = self.edge_act(self.edge(x))
        edge_enhanced = multi_scale * (1 + edge_map)
        b, c, _, _ = x.size()
        chn_avg = self.avg_pool(edge_enhanced).view(b, c)
        chn_att = self.channel_fc(chn_avg).view(b, c, 1, 1)
        chn_out = edge_enhanced * chn_att
        max_pool = torch.max(chn_out, dim=1, keepdim=True)[0]
        mean_pool = torch.mean(chn_out, dim=1, keepdim=True)
        spatial_att = self.spatial_act(self.spatial_conv(torch.cat([max_pool, mean_pool], dim=1)))
        out = chn_out * spatial_att
        out = self.out_conv(out + x)
        boundary_pred = self.boundary_branch(out)
        embedding = self.embedding_branch(out)
        return out, boundary_pred, embedding


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        
        self.b2 = ResBlock(64, 128, 128, stride=2)
        self.b2_1 = ResBlock(128, 128, 128)
        self.b2_2 = ResBlock(128, 128, 128)

        self.b3 = ResBlock(128, 256, 256, stride=2)
        self.b3_1 = ResBlock(256, 256, 256)
        self.b3_2 = ResBlock(256, 256, 256)

        self.b4 = ResBlock(256, 512, 512, stride=2)
        self.b4_1 = ResBlock(512, 512, 512)
        self.b4_2 = ResBlock(512, 512, 512)
        self.b4_3 = ResBlock(512, 512, 512)
        self.b4_4 = ResBlock(512, 512, 512)
        self.b4_5 = ResBlock(512, 512, 512)
        self.bn45 = nn.BatchNorm2d(512)
        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)
        self.bn52 = nn.BatchNorm2d(1024)
        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.3)
        self.IBM = IBM(2048)
        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.5)
        self.bn7 = nn.BatchNorm2d(4096)
        self.not_training = [self.conv1a]
        return

    def forward(self, x):
        return self.forward_as_dict(x)
    def forward_attention(self, x):
        return self.forward_as_dict(x)['at_map']

    def forward_as_dict(self, x):

        x = self.conv1a(x)  #batch*64*224*224
        
        

        x = self.b2(x)      #batch*128*112*112
        x = self.b2_1(x)    #batch*128*112*112
        x = self.b2_2(x)    #batch*128*112*112

        x = self.b3(x)      #batch*256*56*56
        x = self.b3_1(x)    #batch*256*56*56
        x = self.b3_2(x)    #batch*256*56*56

        x = self.b4(x)      #batch*512*28*28
        x = self.b4_1(x)    #batch*512*28*28
        x = self.b4_2(x)    #batch*512*28*28
        x = self.b4_3(x)    #batch*512*28*28
        x = self.b4_4(x)    #batch*512*28*28
        x = self.b4_5(x)    #batch*512*28*28
        b_45 = F.relu(self.bn45(x))
        x, conv4 = self.b5(x, get_x_bn_relu=True)   #x:batch*1024*28*28   conv4:batch*512*28*28
        x = self.b5_1(x)                #x:batch*1024*28*28
        x = self.b5_2(x)                #x:batch*1024*28*28
        b_52 = F.relu(self.bn52(x))
        x, conv5 = self.b6(x, get_x_bn_relu=True)   #x:batch*2048*28*28     conv5:batch*1024*28*28
        x, boundary_pred, embedding = self.IBM(x)
        x = self.b7(x)                  #x:batch*4096*28*28
        at_map = self.bn7(x)            #batch*4096*28*28
        conv6 = F.relu(self.bn7(x))     #batch*4096*28*28
        return b_45, b_52, conv6, boundary_pred, embedding

    def train(self, mode=True):
        super().train(mode)
        for layer in self.not_training:
            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False
            elif isinstance(layer, torch.nn.Module):
                for c in layer.children():
                    c.weight.requires_grad = False
                    if c.bias is not None:
                        c.bias.requires_grad = False
        for layer in self.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False
        return



