# Transformer from Ultralytics YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import  neuron

# SDT v2

# Re-parameterization Convolution
class RepConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        bias=False,
    ):
        super().__init__()
        # hidden_channel = in_channel
        self.conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=1)
        # self.bn = BNAndPadLayer(pad_pixels=1, num_features=in_channel)
        self.bn = nn.BatchNorm2d(in_channel)
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1, groups=in_channel, bias=False),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        # self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.bn(x)
        x = self.conv3x3(x)
        # return self.body(x)
        return x

# 原文Channel MLP，从多步改成单步
class MS_MLP(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = linear_unit(in_features, hidden_features)
        # self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        # self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        # self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.fc1_lif = neuron.LIFNode(tau=2.0, detach_reset=True, v_threshold=1.0)
        # self.fc2 = linear_unit(hidden_features, out_features)
        # self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        # self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        # self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.fc2_lif = neuron.LIFNode(tau=2.0, detach_reset=True, v_threshold=1.0)
        # self.drop = nn.Dropout(0.1)

        self.c_hidden = hidden_features
        self.c_output = out_features

    # 同样没有T
    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x = x.flatten(3)
        x = self.fc1_lif(x)
        # x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_conv(x)
        # x = self.fc1_bn(x).reshape(B, self.c_hidden, N).contiguous()
        x = self.fc1_bn(x)

        x = self.fc2_lif(x)
        # x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_conv(x)
        x = self.fc2_bn(x).reshape(B, C, H, W).contiguous()

        return x

# 原文MS计算self-attention部分
class MS_Attention_RepConv_qkv_id(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        # self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.head_lif = neuron.LIFNode(tau=2.0, detach_reset=True, v_threshold=1.0)

        self.q_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.k_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.v_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        # self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.q_lif = neuron.LIFNode(tau=2.0, detach_reset=True, v_threshold=1.0)

        # self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.k_lif = neuron.LIFNode(tau=2.0, detach_reset=True, v_threshold=1.0)
        # self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.v_lif = neuron.LIFNode(tau=2.0, detach_reset=True, v_threshold=1.0)

        # self.attn_lif = MultiStepLIFNode(
        #     tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
        # )
        self.attn_lif = neuron.LIFNode(tau=2.0, detach_reset=True, v_threshold=0.5)
        self.proj_conv = nn.Sequential(
            RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        # 单步了，没有T了
        B, C, H, W = x.shape
        # N = H * W

        x = self.head_lif(x)
        # kqv，这块没问题，只有v是脉冲
        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        # q = self.q_lif(q).flatten(3)
        # q = (
        #     q.transpose(-1, -2)
        #     .reshape(B, N, self.num_heads, C // self.num_heads)
        #     .permute(0, 1, 3, 2)
        #     .contiguous()
        # )

        # k = self.k_lif(k).flatten(3)
        # k = (
        #     k.transpose(-1, -2)
        #     .reshape(B, N, self.num_heads, C // self.num_heads)
        #     .permute(0, 1, 3, 2)
        #     .contiguous()
        # )

        v = self.v_lif(v).flatten(3)
        # v = (
        #     v.transpose(-1, -2)
        #     .reshape(B, N, self.num_heads, C // self.num_heads)
        #     .permute(0, 1, 3, 2)
        #     .contiguous()
        # )

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale

        # x = x.transpose(3, 4).reshape(B, C, N).contiguous()
        # x = self.attn_lif(x).reshape(B, C, H, W)
        x = self.attn_lif(x).reshape(B, C, H, W)
        # x = x.reshape(B, C, H, W)
        # x = x.flatten(0, 1)
        x = self.proj_conv(x).reshape(B, C, H, W)

        return x

class MS_Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()

        self.attn = MS_Attention_RepConv_qkv_id(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )

        # self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity() #先不要dropout了
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x


# PSA from YOLOv10
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8,
                 attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim*2 + self.head_dim, N).split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (
            (q.transpose(-2, -1) @ k) * self.scale
        )
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x

# Partial Self-Attention from YOLOv10
class PSA(nn.Module):
    def __init__(self, c1, e=0.5):
        super().__init__()
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        # 这块看按上SDT的attention部分
        module_list = [MS_Block(self.c, num_heads=8, mlp_ratio=1.0) for _ in range(4)]
        self.transformer = nn.Sequential(*module_list)
        
        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(
            Conv(self.c, self.c*2, 1),
            Conv(self.c*2, self.c, 1, act=False)
        )
        # self.t = 2
        
    def forward(self, x):
        # for _ in range(self.t):
        #     a, b = self.cv1(x).split((self.c, self.c), dim=1)
        #     b = b + self.attn(b)
        #     b = b + self.ffn(b)
        #     x = self.cv2(torch.cat((a, b), 1))

                # for _ in range(self.t):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.transformer(b)
        # b = b + self.attn(b)
        # b = b + self.ffn(b)
        x = self.cv2(torch.cat((a, b), 1))
        return x

