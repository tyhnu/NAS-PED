import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from mmengine.logging import MMLogger
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mmengine.logging import MMLogger
from mmengine.model import (BaseModule, ModuleList, Sequential, constant_init,
                            normal_init, trunc_normal_init)
from mmengine.model.weight_init import trunc_normal_
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict

from mmdet.registry import MODELS
from ..layers import PatchEmbed, nchw_to_nlc, nlc_to_nchw

def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

def gelu(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.nn.functional, 'gelu'):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0.0)

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # x = nlc2nchw(x, H, W)
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class FFN(nn.Module):
    """
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
            Default: None.
        use_conv (bool): If True, add 3x3 DWConv between two Linear layers.
            Defaults: False.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 # act_cfg=dict(type='GELU'),
                 act_layer=nn.GELU,
                 ffn_drop=0.,
                 drop_path=0.,
                 dropout_layer=None,
                 use_conv=False,
                 scale=False):
        super().__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.use_conv = use_conv
        self.scale = scale


        self.fc1 = nn.Linear(embed_dims, feedforward_channels)

        self.dwconv = DWConv(feedforward_channels)


        self.fc2 = nn.Linear(feedforward_channels, embed_dims)

        self.drop = nn.Dropout(ffn_drop)

        self.act = act_layer()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        out =self.fc1(x)
        out = self.dwconv(out,H,W)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)

        return out

class HybridConvTrans(nn.Module):
    """
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        qkv_bias (bool): enable bias for qkv if True. Default: True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (int): The ratio of spatial reduction of Spatial Reduction
            Attention of PVT. Default: 1.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dim,
                 num_heads,
                 kernel_size=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 qkv_bias=True,
                 qk_scale=None,
                 sr_ratio=1,
                 linear=False,
                 scale=False,
                 change_qkv=True):
        super().__init__()
        assert embed_dim % num_heads == 0, f"dim {embed_dim} should be divided by num_heads {num_heads}."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.linear = linear
        self.kernel_size = kernel_size

        self.fc_scale = scale
        self.change_qkv = change_qkv

        if change_qkv:
            self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
            self.kv = nn.Linear(embed_dim, 2 * embed_dim, bias=qkv_bias)
        else:
            self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
            self.kv = nn.Linear(embed_dim, 2 * embed_dim, bias=qkv_bias)


        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        self.act = nn.GELU()


        if self.kernel_size == 3:
            self.fc = nn.Conv2d(3 * self.num_heads, self.kernel_size * self.kernel_size, kernel_size=1,
                                stride=1)
            self.dep_conv = nn.Conv2d(9*embed_dim//self.num_heads, embed_dim,kernel_size=3, stride=1, bias=True,
                                        groups=embed_dim//self.num_heads, padding=1)
            self.conv_norm = nn.LayerNorm(embed_dim)
        elif self.kernel_size == 5:
            self.fc = nn.Conv2d(3 * self.num_heads, self.kernel_size * self.kernel_size, kernel_size=1,
                                stride=1)
            self.dep_conv = nn.Conv2d(25*embed_dim//self.num_heads, embed_dim,kernel_size=5, stride=1, bias=True,
                                        groups=embed_dim//self.num_heads, padding=2)
            self.conv_norm = nn.LayerNorm(embed_dim)


        else:
            self.attn_drop = nn.Dropout(attn_drop)
            if not linear:
                if sr_ratio > 1:
                    self.sr = nn.Conv2d(embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio)
                    self.norm = nn.LayerNorm(embed_dim)
            else:
                self.pool = nn.AdaptiveAvgPool2d(7)
                self.sr = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1)
                self.norm = nn.LayerNorm(embed_dim)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def reset_parameters_3(self):
        kernel = torch.zeros(9, 3, 3)
        for i in range(9):
            kernel[i, i//3, i%3] = 1.
        kernel = kernel.squeeze(0).repeat(self.embed_dim, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = zeros(self.dep_conv.bias)

    def reset_parameters_5(self):
        kernel = torch.zeros(25, 5, 5)
        for i in range(25):
            kernel[i, i//5, i%5] = 1.
        kernel = kernel.squeeze(0).repeat(self.embed_dim, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = zeros(self.dep_conv.bias)

    def forward(self, x, H, W):
        B, N, C = x.shape

        q = self.q(x)
        kv_ = self.kv(x).reshape(B, N, 2*self.num_heads, -1).permute(0, 2, 1, 3).contiguous()

        q = q.reshape(B, N, self.num_heads, -1)
        _,_,_,d=q.shape
        q = q.permute(0, 2, 1, 3).contiguous()

        if self.kernel_size==3:  ##Conv3 output
            qkv = torch.cat((q, kv_), dim=1)
            f_all = qkv.reshape(B, H * W, 3 * self.num_heads, -1).permute(0, 2, 1,
                                                                                 3).contiguous()  # B, 3*nhead, H*W, C//nhead
            f_conv = self.fc(f_all).permute(0, 3, 1, 2).contiguous()
            f_conv = f_conv.reshape(B, -1, H, W).contiguous()  # B, 9*C//nhead, H, W
            out_conv = self.dep_conv(f_conv).permute(0, 2, 3, 1).reshape(B, N, -1).contiguous() # B, H, W, C
            out_conv = self.conv_norm(out_conv)
            out_conv3 = self.act(out_conv)

        elif self.kernel_size==5:  ##Conv5 output
            qkv = torch.cat((q, kv_), dim=1)
            f_all = qkv.reshape(B, H * W, 3 * self.num_heads, -1).permute(0, 2, 1,
                                                                                 3).contiguous()  # B, 3*nhead, H*W, C//nhead
            f_conv = self.fc(f_all).permute(0, 3, 1, 2).contiguous()
            f_conv = f_conv.reshape(B, -1, H, W).contiguous()  # B, 25*C//nhead, H, W
            out_conv = self.dep_conv(f_conv).permute(0, 2, 3, 1).reshape(B, N, -1).contiguous()  # B, H, W, C
            out_conv = self.conv_norm(out_conv)
            out_conv5 = self.act(out_conv)

        else:  ##MHSA output (Spatial Reduction of PVT)
            if not self.linear:
                if self.sr_ratio > 1:
                    x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                    x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                    x_ = self.norm(x_)
                    kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, d).permute(2, 0, 3, 1, 4)
                else:
                    kv = self.kv(x).reshape(B, -1, 2, self.num_heads, d).permute(2, 0, 3, 1, 4)

            else:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                x_ = self.act(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, d).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]


            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            out_mhsa = (attn @ v).transpose(1, 2).reshape(B, N, -1)


        if self.kernel_size == 3:
            x=out_conv3

        elif self.kernel_size == 5:
            x=out_conv5
        else:
            x=out_mhsa

        # projection output
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class EncoderLayer(nn.Module):
    """
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed.
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): stochastic depth rate. Default: 0.0.
        qkv_bias (bool): enable bias for qkv if True.
            Default: True.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (int): The ratio of spatial reduction of Spatial Reduction
            Attention of PVT. Default: 1.
        use_conv_ffn (bool): If True, use Convolutional FFN to replace FFN.
            Default: False.
        init_cfg (dict, optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 dim,
                 num_heads,
                 kernel_size=None,
                 mlp_ratio=4.,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 qkv_bias=True,
                 qk_scale=None,
                 pre_norm=True,
                 sr_ratio=1,
                 scale=False,
                 linear=False,
                 use_conv_ffn=False):
        super().__init__()
        self.embed_dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.normalize_before = pre_norm
        self.use_conv_ffn=use_conv_ffn

        # the configs of current sampled arch
        self.scale = scale
        self.is_identity_layer = None

        self.attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.ffn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.attn = HybridConvTrans(
            embed_dim=dim, kernel_size=kernel_size,
            num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop, qkv_bias=qkv_bias, qk_scale=qk_scale,
            sr_ratio=sr_ratio, linear=linear)


        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.linear = linear
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W,):
        if self.is_identity_layer:
            return x
        x = x + self.drop_path(self.attn(self.attn_layer_norm(x), H, W))
        x = x + self.drop_path(self.ffn(self.ffn_layer_norm(x), H, W))
        return x



class Patchembed(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 stride=4,
                 in_chans=3,
                 embed_dim=768,
                 scale=False,
                 ):
        super(Patchembed, self).__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.super_embed_dim = embed_dim
        self.scale = scale

        self.sample_embed_dim = None
        self.sampled_weight = None
        self.sampled_bias = None
        self.sampled_scale = None
        self.sampled_norm = {}

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W



@MODELS.register_module()
class NAS_SuperNet(BaseModule):
    def __init__(self,
                pretrain_img_size=224,
                in_chans=3,
                embed_dims=[64, 192, 384, 640],
                num_stages=4,
                depths=[10, 10, 40, 10],
                kernel_size=None,
                num_heads=[1, 2, 5, 8],
                mlp_ratio=[4, 4, 4, 4],
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                norm_layer=nn.LayerNorm,
                drop_path_rate=0.1,
                linear=False,
                num_classes=1,
                pretrained=None,
                init_cfg=None,
                frozen_stage=2):
        super().__init__(init_cfg=init_cfg)
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        self.num_stages = num_stages
        self.sr_ratios = [8, 4, 2, 1]  # Spatial Reduction of PVT
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.depths = depths
        self.blocks = nn.ModuleList()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(self.num_stages):
            patch_embed = Patchembed(img_size=pretrain_img_size if i == 0 else pretrain_img_size // (2 ** (i + 1)),
                                        patch_size=7 if i == 0 else 3,
                                        stride=4 if i == 0 else 2,
                                        in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                        embed_dim=embed_dims[i])

            blocks = nn.ModuleList([EncoderLayer(
                dim=embed_dims[i], num_heads=num_heads[i][j], kernel_size=kernel_size[i][j], mlp_ratio=mlp_ratio[i][j], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=self.sr_ratios[i], linear=linear) for j in range(depths[i])])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", blocks)
            setattr(self, f"norm{i + 1}", norm)

        self.init_weights()
        self.freeze_stages(frozen_stage=frozen_stage)


    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, 0, math.sqrt(2.0 / fan_out))
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            checkpoint = _load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            logger.warn(f'Load pre-trained model for '
                        f'{self.__class__.__name__} from original repo')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            load_state_dict(self, state_dict, strict=False, logger=logger)


    def freeze_stages(self,frozen_stage=2):
        for i in range(frozen_stage):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            patch_embed.requires_grad = False
            block.requires_grad = False
            norm.requires_grad = False


    def forward_features(self, x):
        B = x.shape[0]
        outs=[]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            hw_shape = (H, W)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)

            x = nlc_to_nchw(x,hw_shape)
            outs.append(x)
            # if i in self.out_indices:
            #     outs.append(x)
        return outs
        # return x.mean(dim=1)



    def forward(self, x):
        x = self.forward_features(x)
        return x



def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict

@MODELS.register_module()
class NAS_PED_Small(NAS_SuperNet):
    def __init__(self, **kwargs):
        super(NAS_PED_Small, self).__init__(
            embed_dims=[64, 128, 320, 512],
            depths=[4, 5, 7, 3],
            num_heads=[[1, 1, 1, 1], [2, 2, 2, 2, 2], [5, 5, 5, 5, 5, 5, 5], [8, 8, 8]],
            mlp_ratio=[[7.5, 8.0, 8.0, 7.5], [8.5, 7.5, 8.0, 7.5, 8.0],[4.0, 4.5, 4.5, 3.5, 3.5, 4.0, 3.5], [3.5, 3.5, 4.0]],
            kernel_size=[[1, 1, 3, 1], [1, 3, 1, 1, 1], [1, 1, 3, 5, 3, 1, 1], [1, 1, 1]],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs)


@MODELS.register_module()
class NAS_PED_Tiny(NAS_SuperNet):
    def __init__(self, **kwargs):
        super(NAS_PED_Tiny, self).__init__(
            embed_dims=[64, 128, 256, 448],
            depths=[2, 3, 2, 3],
            num_heads=[[1, 1], [2, 2, 2], [4, 4], [7, 7, 7]],
            mlp_ratio= [[8.5, 7.5], [7.5, 7.5, 8.5], [3.5, 4.5], [4.0, 3.5, 4.0]],
            kernel_size=[[3, 1], [1, 1, 1], [1, 1], [3, 1, 1]],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs)

@MODELS.register_module()
class NAS_PED_Base(NAS_SuperNet):
    def __init__(self, **kwargs):
        super(NAS_PED_Base, self).__init__(
            embed_dims=[64, 192, 320, 448],
            depths= [3, 3, 15, 5],
            num_heads=[[1, 1, 1], [3, 3, 3], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,5, 5, 5], [7, 7, 7,7,7]],
            mlp_ratio=[[8.0, 8.5, 8.0], [8.5, 7.5, 8.5], [4.5, 4.0, 4.0, 4.5, 4.0, 3.5, 4.0, 3.5, 4.5, 3.5, 4.5, 4.5, 3.5, 4.5, 4.5],[3.5, 4.0, 4.0, 3.5, 4.5]],
            kernel_size=[[1, 1, 3], [1, 1, 1], [5, 1, 5, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 5], [5, 1, 1, 1, 3]],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs)








