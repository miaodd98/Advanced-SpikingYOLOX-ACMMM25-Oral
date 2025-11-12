#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
# Adding spiking neurons to replace activation functions by SuperCarKing https://github.com/miaodd98

import torch
from spikingjelly.activation_based import neuron, surrogate, layer
from torch import nn
from .ffcplus import FFCResnetBlock
from .transformer import PSA
from .convs import C2f, C2fAttn, Conv2

class SignedIFNode(neuron.IFNode):
    def __init__(self,surrogate_function: surrogate.ATan(),detach_reset: bool = False):
        super().__init__()
        self.pos_cnt = 0
        self.neg_threshold = torch.tensor(1e-3)
        self.neg_spike = torch.tensor(-1.0)

    # 脉冲发放
    def neuronal_fire(self):
        # 负脉冲
        if self.pos_cnt > 0 and self.v <= self.neg_threshold:
            self.spike = self.neg_spike
        # 负脉冲以外情况
        else:
            self.spike = self.surrogate_function(self.v - self.v_threshold)
            self.pos_cnt += 1
        return self.spike
    
    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x + self.surrogate_function(self.pos_cnt - torch.tensor(1)) * self.surrogate_function(self.v - self.neg_threshold)
    
    # 脉冲重置
    def neuronal_reset(self, spike):
        self.pos_cnt = 0
        return super().neuronal_reset(spike)

# 加负脉冲部分，看看要不要留着
class LIFNode_with_negative_impulse(neuron.LIFNode):
    # 脉冲发放
    def neuronal_fire(self):
        self.spike = 2 * (self.surrogate_function(self.v - self.v_threshold) - 0.5)     # 比方说这样它就成了输出为[-1, 1]两种
        return self.spike
    # 脉冲重置，现在无脉冲spike值为-1，有脉冲为1，现在的处理就是把-1和1这块改回去了
    def neuronal_reset(self, spike):
        if self.detach_reset:
            # spike_d = spike.detach()
            spike_d = spike.detach() / 2 + 0.5
        else:
            # spike_d = spike
            spike_d = spike / 2 + 0.5

        if self.v_reset is None:
            # soft reset
            self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold)

        else:
            # hard reset
            self.v = self.jit_hard_reset(self.v, spike_d, self.v_reset)

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

# 激活函数
def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "spiking":
        # module = neuron.LIFNode(surrogate_function=surrogate.ATan())
        # module = LIFNode_with_negative_impulse(surrogate_function=surrogate.ATan())
        module = SignedIFNode(surrogate_function=surrogate.ATan())
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

# Focus这里不是只有一个卷积层了，直接改成Conv-BN这里，也不要激活函数了
# class Focus(nn.Module):
#     def __init__(self, in_channels, out_channels, ksize=1, stride=1):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size=ksize, stride=stride)
#         # self.bn = nn.BatchNorm2d(out_channels)
#         self.silu = SiLU()

#     def forward(self, x):
#         patch_top_left  = x[...,  ::2,  ::2]
#         patch_bot_left  = x[..., 1::2,  ::2]
#         patch_top_right = x[...,  ::2, 1::2]
#         patch_bot_right = x[..., 1::2, 1::2]
#         x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1,)
#         return self.silu(self.conv(x))    # return self.bn(self.conv(x))
    
class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        patch_top_left  = x[...,  ::2,  ::2]
        patch_bot_left  = x[..., 1::2,  ::2]
        patch_top_right = x[...,  ::2, 1::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1,)
        return self.conv(x)


# 这个BaseConv带激活函数了，激活函数可以选用传统函数还是SNN脉冲
# 如果不需要激活函数，act部分设置为None
class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad         = (ksize - 1) // 2
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.activation = act if isinstance(act, str) else None
        self.act    = get_activation(act, inplace=True) if act else nn.Identity()

    def forward(self, x):
        if self.activation == None:
            return self.bn(self.conv(x))
        else:
            return self.act(self.bn(self.conv(x)))
        

class BaseConvSNN(nn.Module):       # 使用SpikingJelly的BaseConv
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="spiking"):
        super().__init__()
        pad         = (ksize - 1) // 2
        self.conv   = layer.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn     = layer.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        # self.instance = nn.InstanceNorm2d(out_channels, eps=0.001, momentum=0.03)
        # self.bn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act    = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
        # return self.act(self.instance(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

   
# 这里看一下DW和PWConv的顺序
# depth-wise conv
class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act,)
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

# PW-DW conv
class PWDWConvSNN(nn.Module):         
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="spiking"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act,)
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        # 先pwconv在dwconv
        x = self.pconv(x)
        x = self.dconv(x)
        return x

# 替换为SPPF的改进版，从yolo v10中移植，出自yolo v5
class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=5, activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1      = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        # self.m          = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        self.m  = nn.MaxPool2d(kernel_size=kernel_sizes, stride=1, padding=kernel_sizes // 2)
        # conv2_channels  = hidden_channels * (len(kernel_sizes) + 1)
        conv2_channels  = hidden_channels * 4
        self.conv2      = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.m(x1)
        x3 = self.m(x2)
        x4 = self.m(x3)
        # 过三遍max pool？
        x = self.conv2(torch.cat((x1, x2, x3, x4), dim=1))
        # x = self.conv2(torch.cat((x, x1, x2, self.m(x3)), dim=1))
        return x

#--------------------------------------------------#
#   残差结构的构建，小的残差结构，都换成SNN-卷积模块
#--------------------------------------------------#
    
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act="silu"):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = PWDWConvSNN if depthwise else BaseConv
        #--------------------------------------------------#
        #   利用1x1卷积进行通道数的缩减。缩减率一般是50%
        #--------------------------------------------------#
        # self.conv1 = BaseConvSNN(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        #--------------------------------------------------#
        #   利用3x3卷积进行通道数的拓张。并且完成特征提取
        #--------------------------------------------------#
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y
    

# Adding activation/spiking on C2f
class C2f_ACT(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, act="silu"):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv2(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv2((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.act    = get_activation(act, inplace=True) if act else nn.Identity()

    def forward(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act="silu"):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  
        #--------------------------------------------------#
        #   主干部分的初次卷积
        #--------------------------------------------------#
        # self.conv1  = BaseConvSNN(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv1  = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        #--------------------------------------------------#
        #   大的残差边部分的初次卷积
        #--------------------------------------------------#
        # self.conv2  = BaseConvSNN(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2  = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        #-----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        #-----------------------------------------------#
        # self.conv3  = BaseConvSNN(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        self.conv3  = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        #--------------------------------------------------#
        #   根据循环的次数构建上述Bottleneck残差结构，中间层换用SEWResBlock
        #--------------------------------------------------#
        # module_list = [SEWBottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in range(n)]
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in range(n)]
        self.m      = nn.Sequential(*module_list)

    def forward(self, x):
        #-------------------------------#
        #   x_1是主干部分
        #-------------------------------#
        x_1 = self.conv1(x)
        #-------------------------------#
        #   x_2是大的残差边部分
        #-------------------------------#
        x_2 = self.conv2(x)

        #-----------------------------------------------#
        #   主干部分利用残差结构堆叠继续进行特征提取
        #-----------------------------------------------#
        x_1 = self.m(x_1)
        #-----------------------------------------------#
        #   主干部分和大的残差边部分进行堆叠
        #-----------------------------------------------#
        x = torch.cat((x_1, x_2), dim=1)
        #-----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        #-----------------------------------------------#
        return self.conv3(x)


# 修改为适应SNN-卷积模块的部分    
# 这块重点修改，改结构看看
class CSPFFCLayer_new(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act="spiking",ffc=False):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  
        #--------------------------------------------------#
        #   主干部分的初次卷积
        #--------------------------------------------------#
        self.conv1  = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        #--------------------------------------------------#
        #   大的残差边部分的初次卷积
        #--------------------------------------------------#
        self.conv2  = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        #-----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        #-----------------------------------------------#
        self.conv3  = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        #--------------------------------------------------#
        #   根据循环的次数构建上述Bottleneck残差结构，中间层换用SEWResBlock
        #--------------------------------------------------#
        
        module_list = [FFCResnetBlock(hidden_channels) for _ in range(n)] if ffc else [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in range(n)]
        
        self.m      = nn.Sequential(*module_list)

    def forward(self, x):
        #-------------------------------#
        #   x_1是主干部分
        #-------------------------------#
        x_1 = self.conv1(x)
        #-------------------------------#
        #   x_2是大的残差边部分
        #-------------------------------#
        x_2 = self.conv2(x)

        #-----------------------------------------------#
        #   主干部分利用残差结构堆叠继续进行特征提取
        #-----------------------------------------------#
        x_1 = self.m(x_1)
        #-----------------------------------------------#
        #   主干部分和大的残差边部分进行堆叠
        #-----------------------------------------------#
        x = torch.cat((x_1, x_2), dim=1)
        #-----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        #-----------------------------------------------#
        return self.conv3(x)
    
class CSPFFCLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act="spiking"):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  
        #--------------------------------------------------#
        #   主干部分的初次卷积
        #--------------------------------------------------#
        self.conv1  = BaseConvSNN(in_channels, hidden_channels, 1, stride=1, act=act)
        #--------------------------------------------------#
        #   大的残差边部分的初次卷积
        #--------------------------------------------------#
        self.conv2  = BaseConvSNN(in_channels, hidden_channels, 1, stride=1, act=act)
        #-----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        #-----------------------------------------------#
        self.conv3  = BaseConvSNN(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        #--------------------------------------------------#
        #   根据循环的次数构建上述Bottleneck残差结构，中间层换用SEWResBlock
        #--------------------------------------------------#
        module_list = [FFCResnetBlock(hidden_channels) for _ in range(n)]
        self.m      = nn.Sequential(*module_list)

    def forward(self, x):
        #-------------------------------#
        #   x_1是主干部分
        #-------------------------------#
        x_1 = self.conv1(x)
        #-------------------------------#
        #   x_2是大的残差边部分
        #-------------------------------#
        x_2 = self.conv2(x)

        #-----------------------------------------------#
        #   主干部分利用残差结构堆叠继续进行特征提取
        #-----------------------------------------------#
        x_1 = self.m(x_1)
        #-----------------------------------------------#
        #   主干部分和大的残差边部分进行堆叠
        #-----------------------------------------------#
        x = torch.cat((x_1, x_2), dim=1)
        #-----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        #-----------------------------------------------#
        return self.conv3(x)



class SPPBottleneckSNN(nn.Module):         # 使用SpikingJelly的SPP
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="spiking"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1      = BaseConvSNN(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m          = nn.ModuleList([layer.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels  = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2      = BaseConvSNN(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x

# 五层Darknet，正在修改，现在看来在最后送到检测头需要经过BN，而非直接将脉冲送到检测头
# C2f可以提点
class CSPDarknet(nn.Module):
    def __init__(self, dep_mul, wid_mul, out_features=("dark3", "dark4", "dark5"), depthwise=False, act="silu"):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        # Conv = DWConvSNN if depthwise else BaseConvSNN
        Conv = DWConv if depthwise else BaseConv
        self.act    = get_activation(act, inplace=True)
        self.act_snn = get_activation("spiking", inplace=False)

        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        #-----------------------------------------------#
        base_channels   = int(wid_mul * 64)  # 64
        base_depth      = max(round(dep_mul * 3), 1)  # 3
        
        #-----------------------------------------------#
        #   利用focus网络结构进行特征提取
        #   640, 640, 3 -> 320, 320, 12 -> 320, 320, 64
        #-----------------------------------------------#
        # self.stem = FocusSNN(3, base_channels, ksize=3, act=act)
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        #-----------------------------------------------#
        #   完成卷积之后，320, 320, 64 -> 160, 160, 128
        #   完成CSPlayer之后，160, 160, 128 -> 160, 160, 128
        #-----------------------------------------------#
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise, act=act),
        )

        #-----------------------------------------------#
        #   完成卷积之后，160, 160, 128 -> 80, 80, 256
        #   完成CSPlayer之后，80, 80, 256 -> 80, 80, 256
        #-----------------------------------------------#
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act),
            # CSPFFCLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        #-----------------------------------------------#
        #   完成卷积之后，80, 80, 256 -> 40, 40, 512
        #   完成CSPlayer之后，40, 40, 512 -> 40, 40, 512
        #-----------------------------------------------#
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act),
            # CSPFFCLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        #-----------------------------------------------#
        #   完成卷积之后，40, 40, 512 -> 20, 20, 1024
        #   完成SPP之后，20, 20, 1024 -> 20, 20, 1024
        #   完成CSPlayer之后，20, 20, 1024 -> 20, 20, 1024
        #-----------------------------------------------#
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneckSNN(base_channels * 16, base_channels * 16, activation=act),
            # CSPFFCLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise, act="spiking", ffc=True),
            CSPFFCLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise, act="spiking"),

        )
        self.psa = PSA(base_channels * 16, e=0.5)
        self.d5down = Conv(base_channels * 32, base_channels * 16, 1, 1)


    def forward(self, x):
        outputs = {}
        # 第一步Focus卷积和激活函数保留
        x = self.stem(x)
        # x = self.act(x)
        outputs["stem"] = x
        # 后面的开始大改
        x = self.dark2(x)
        # x = self.act(x)
        outputs["dark2"] = x
        #-----------------------------------------------#
        #   dark3的输出为80, 80, 256，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark3(x)
        # x = self.act(x)
        outputs["dark3"] = x
        #-----------------------------------------------#
        #   dark4的输出为40, 40, 512，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark4(x)
        # x = self.act(x)
        outputs["dark4"] = x
        #-----------------------------------------------#
        #   dark5的输出为20, 20, 1024，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark5(x)
        x1 = self.psa(x)
        x2 = self.d5down(torch.cat((x, x1), dim=1))
        outputs["dark5"] = x2
        return {k: v for k, v in outputs.items() if k in self.out_features}


if __name__ == '__main__':
    print(CSPDarknet(1, 1))