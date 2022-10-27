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
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)


        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):

            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:

            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, sequence_len, c, num_patches=8, dim=512, depth=6, heads=4, mlp_dim=256, dim_head = 64, pool = 'mean', dropout = 0.1,emb_dropout = 0.1):
        super().__init__()
        self.to_patch_embedding_1 = nn.Sequential(
            Rearrange('b c (m n)-> b n (c m)', n = num_patches,c=c),
            nn.Linear(sequence_len, dim),
        )
        self.to_patch_embedding_2 = nn.Sequential(
            #Rearrange('b c (m n)-> b n (c m)', n = num_patches,c=c),
            nn.Linear(sequence_len, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim)) 
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) 

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.flag_em = flag_em
        self.flag_mlp = flag_mlp
        self.to_latent = nn.Identity()

        self.mlp_head_1 = nn.Sequential(
            nn.LayerNorm(dim),
            #nn.Linear(dim, sequence_len),
            #Rearrange('b n (c m)-> b c (m n)', n = num_patches, c=c)
        )
        self.mlp_head_2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, sequence_len),
            Rearrange('b n (c m)-> b c (m n)', n = num_patches, c=c)
        )


    def forward(self, data):
        if self.flag_em:
            x = self.to_patch_embedding_1(data)
        else:
            x = self.to_patch_embedding_2(data)
        b, n, _ = x.shape  # shape (b, n, 1024)

        x += self.pos_embedding[:, :]
        x = self.dropout(x)

        x = self.transformer(x)


        x = self.to_latent(x)
        if self.flag_mlp:
            out = self.mlp_head_2(x)
        else:
            out = self.mlp_head_1(x)
        return out


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


class ResNet_CTTT(nn.Module):
    def __init__(self, block=Bottleneck, layers=[2,2,2,2], channels = 16, num_classes=2):
        self.inplanes = channels
        super(ResNet_CTTT, self).__init__()
        self.conv1 = nn.Conv1d(1, channels, kernel_size=7, stride=2, padding=3,
                        bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, channels, layers[0])
        self.layer2 = self._make_layer(block, channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels * 8, layers[3], stride=2)   # different

        self.ViT_1 = ViT(sequence_len = channels*128,c = channels*4,num_patches=8, flag_em=True,flag_mlp=False,dim=512)
        self.ViT_2 = ViT(sequence_len = channels*32,c = channels*4,num_patches=8,flag_em=False,flag_mlp=False,dim=256)
        self.ViT_3 = ViT(sequence_len = channels*16,c = channels//2,num_patches=8,flag_em=False,flag_mlp=True,dim=128)
        self.conv2 = nn.Conv1d(channels * 2, channels, kernel_size=3, stride=2, padding=1,
                        bias=False)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(channels * 8, num_classes)
        self.linear1 = nn.Linear(channels // 2, channels//2)
        self.linear2 = nn.Linear(channels//2, num_classes)

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
        x_t1 = self.ViT_1(x)
        x_t2 = self.ViT_2(x_t1)
        x_t3 = self.ViT_3(x_t2)

        #x_t2 = self.conv2(x_t2)
        x_ = self.avgpool(x_t3)
        x_ = x_.view(x_.size(0), -1)
        x_ = self.linear1(x_)
        out = self.linear2(x_)

        return out


if __name__ == '__main__':
    pre_model = ResNet_CTTT(Bottleneck, layers=[2,2,2,2]).cuda()
    summary(pre_model,(1, 1024))
    img = torch.randn(1, 1, 1024)
    out = pre_model(img)

