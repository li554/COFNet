import numbers

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class DeepWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DeepWiseConv, self).__init__()
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.dw_conv = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                 padding=padding, groups=out_channels, bias=bias)

    def forward(self, x):
        x = self.pw_conv(x)
        x = self.dw_conv(x)
        return x


# class CrossAttention(nn.Module):
#     def __init__(self, dim, num_heads, bias):
#         super(CrossAttention, self).__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
#         self.q = DeepWiseConv(dim, dim, bias=bias)
#         self.k = DeepWiseConv(dim, dim, bias=bias)
#         self.v = DeepWiseConv(dim, dim, bias=bias)
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#
#     def forward(self, x, y, z):
#         b, c, h, w = x.shape
#         # q, k, v = self.q(x), self.k(y), self.v(z)
#         # q, k, v = x, y, z
#         q, k, v = x + self.q(x), y + self.k(y), z + self.v(z)
#         q = rearrange(q, 'b (head c) h w -> b head c (h w)',
#                       head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)',
#                       head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)',
#                       head=self.num_heads)
#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)
#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)
#         out = (attn @ v)
#         out = rearrange(out, 'b head c (h w) -> b (head c) h w',
#                         head=self.num_heads, h=h, w=w)
#         out = self.project_out(out)
#         return out


class CrossConvTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(CrossConvTransformerBlock, self).__init__()

        self.norm1_x = LayerNorm(dim, LayerNorm_type)
        self.norm1_y = LayerNorm(dim, LayerNorm_type)
        self.norm1_z = LayerNorm(dim, LayerNorm_type)
        self.attn1 = CrossAttention(dim, num_heads, bias)
        self.attn2 = CrossAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, y, z, p="r_only"):
        # x: mask, y: infrared, z: visible light
        norm_x = self.norm1_x(x)
        norm_y = self.norm1_y(y)
        norm_z = self.norm1_z(z)
        if p == "r_only":
            y = y + self.attn1(norm_x, norm_y, norm_z) + self.attn2(norm_x, norm_z, norm_y)
        elif p == "v_only":
            y = z + self.attn1(norm_x, norm_y, norm_z) + self.attn2(norm_x, norm_z, norm_y)
        elif p == "v_r":
            y = z + y + self.attn1(norm_x, norm_y, norm_z) + self.attn2(norm_x, norm_z, norm_y)
        else:
            y = self.attn1(norm_x, norm_y, norm_z) + self.attn2(norm_x, norm_z, norm_y)
        y = y + self.ffn(self.norm2(y))
        return y


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = DeepWiseConv(dim, dim, bias=bias)
        self.k = DeepWiseConv(dim, dim, bias=bias)
        self.v = DeepWiseConv(dim, dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y, z):
        b, c, h, w = x.shape
        # q, k, v = self.q(x), self.k(y), self.v(z)
        # q, k, v = x, y, z
        # patch_size = 2 if h > 32 else 1
        patch_size = 1
        q, k, v = x + self.q(x), y + self.k(y), z + self.v(z)
        # rearrange q, k, v to from (b, c, h, w) to (b, head, c//head*h//patch_size*w//patch_size, patch_size*patch_size)
        q = rearrange(q, 'b (head c) (h p1) (w p2) -> b head (c p1 p2) (h w)', head=self.num_heads, p1=patch_size,
                      p2=patch_size)
        k = rearrange(k, 'b (head c) (h p1) (w p2) -> b head (c p1 p2) (h w)', head=self.num_heads, p1=patch_size,
                      p2=patch_size)
        v = rearrange(v, 'b (head c) (h p1) (w p2) -> b head (c p1 p2) (h w)', head=self.num_heads, p1=patch_size,
                      p2=patch_size)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head (c p1 p2) (h w) -> b (head c) (h p1) (w p2)', head=self.num_heads,
                        p1=patch_size, p2=patch_size, h=h // patch_size, w=w // patch_size)
        out = self.project_out(out)
        return out


class AttentionBase(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False, ):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out


class NewCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(NewCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = DeepWiseConv(dim, dim, bias=bias)
        self.k = DeepWiseConv(dim, dim, bias=bias)
        self.v = DeepWiseConv(dim, dim, bias=bias)
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y, z):
        b, c, h0, w0 = x.shape
        x, y, z = self.avg_pool(x), self.avg_pool(y), self.avg_pool(z)
        b, c, h, w = x.shape
        q, k, v = self.q(y), self.k(z), self.v(z)
        q = rearrange(q, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        x = rearrange(x, 'b (head c) h w -> b head (h w) c', head=1)
        q = torch.nn.functional.normalize(q, dim=-1)
        # q = q * x
        # q = torch.nn.functional.normalize(q, dim=-2)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        out = F.interpolate(input=out, size=(h0, w0), mode='bilinear', align_corners=False)
        return out


class NewTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(NewTransformerBlock, self).__init__()
        self.norm1_y = LayerNorm(dim, LayerNorm_type)
        self.norm1_z = LayerNorm(dim, LayerNorm_type)
        self.attn1 = NewCrossAttention(dim, num_heads, bias)
        self.attn2 = NewCrossAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.alpha = nn.Parameter(torch.FloatTensor([1]), requires_grad=True)
        self.beta = nn.Parameter(torch.FloatTensor([1]), requires_grad=True)

    def forward(self, x, y, z, p="sam"):
        norm_y = self.norm1_y(y)
        norm_z = self.norm1_z(z)
        y = self.alpha * y + self.beta * z + self.attn1(x, norm_y, norm_z) + self.attn2(x, norm_z, norm_y)
        # y = self.alpha * y + self.beta * z
        y = y + self.ffn(self.norm2(y))
        return y
