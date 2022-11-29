import torch.nn as nn
import torch
from torchsummary import summary

from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        # 先LN
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        # 前向传播
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    # attention
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads  # 计算最终进行全连接操作时输入神经元的个数
        project_out = not (heads == 1 and dim_head == dim)  # 多头注意力并且输入和输出维度相同时为True

        self.heads = heads  # 多头注意力中“头”的个数
        self.scale = dim_head ** -0.5  # 缩放操作，论文 Attention is all you need 中有介绍

        self.attend = nn.Softmax(dim = -1)  # 初始化一个Softmax操作
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)  # 对Q、K、V三组向量先进性线性操作

        # 线性全连接，如果不是多头或者输入输出维度不相等，进行空操作
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads  # 获得输入x的维度和多头注意力的“头”数
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # 先对Q、K、V进行线性操作，然后chunk乘三三份
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)  # 整理维度，获得Q、K、V

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # Q, K 向量先做点乘，来计算相关性，然后除以缩放因子

        attn = self.attend(dots)  # 做Softmax运算

        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # Softmax运算结果与Value向量相乘，得到最终结果
        out = rearrange(out, 'b h n d -> b n (h d)')  # 重新整理维度
        return self.to_out(out)  # 做线性的全连接操作或者空操作（空操作直接输出out）


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])  # Transformer包含多个编码器的叠加
        for _ in range(depth):
            # 编码器包含两大块：自注意力模块和前向传播模块
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),  # 多头自注意力模块
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))  # 前向传播模块
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            # 自注意力模块和前向传播模块都使用了残差的模式
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, sequence_len, c, num_patches=20, dim=512, depth=6, heads=4, mlp_dim=256, dim_head = 64, pool = 'mean', dropout = 0.1,emb_dropout = 0.1):
        super().__init__()
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (m n)-> b n (c m)', n = num_patches,c=c),
            nn.Linear(sequence_len, dim),  # 对分割好的图像块进行线性处理（全连接），输入维度为每一个小块的所有像素个数，输出为dim（函数传入的参数）
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))  # 位置编码，获取一组正态分布的数据用于训练
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 分类令牌，可训练

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)  # Transformer模块

        self.pool = pool
        self.to_latent = nn.Identity()  # 占位操作

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),  # 正则化
            nn.Linear(dim, sequence_len),
            Rearrange('b n (c m)-> b c (m n)', n = num_patches, c=c)
        )


    def forward(self, data):
        x = self.to_patch_embedding(data)  # 切块操作，shape (b, n, dim)，b为批量，n为切块数目，dim为最终线性操作时输入的神经元个数
        b, n, _ = x.shape  # shape (b, n, 1024)

        #cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)  # 分类令牌，将self.cls_token（形状为1, 1, dim）赋值为shape (b, 1, dim)
        #x = torch.cat((cls_tokens, x), dim=1)  # 将分类令牌拼接到输入中，x的shape (b, n+1, 1024)
        x += self.pos_embedding[:, :]  # 进行位置编码，shape (b, n+1, 1024)
        x = self.dropout(x)

        x = self.transformer(x)  # transformer操作

        #x = x[:, 1:].mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)  # 线性输出

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm1d(planes)
        
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, # change
                    padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_50_CCCT(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3,4,6,3], channels = 64, num_classes=2):
        self.inplanes = channels
        super(ResNet_50_CCCT, self).__init__()
        self.conv1 = nn.Conv1d(1, channels, kernel_size=7, stride=2, padding=3,
                        bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, channels, layers[0])
        self.layer2 = self._make_layer(block, channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels * 8, layers[3], stride=2)   # different

        self.ViT = ViT(sequence_len = channels*128,c = channels*16,num_patches=8,dim=416)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear1 = nn.Linear(channels * 16, channels * 8)
        self.linear2 = nn.Linear(channels * 8, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
            nn.Conv1d(self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_t = self.ViT(x)
        x = x + x_t 
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        out = self.linear2(x)

        return out


if __name__ == '__main__':
    pre_model = ResNet_50_CCCT(Bottleneck, layers=[3,4,6,3],channels = 16).cuda()
    summary(pre_model,(1,8*128))
    img = torch.randn(1, 1, 20 * 128)
    out = pre_model(img)

