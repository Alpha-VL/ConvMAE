# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# Swin Transformer: https://github.com/microsoft/Swin-Transformer
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import collections
import logging
import math
from collections import defaultdict, OrderedDict
from functools import partial
from itertools import repeat
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone
from detectron2.utils.logger import setup_logger
from einops import rearrange
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.layers import variance_scaling_

__all__ = ["BenchmarkingViTDet"]


def trunc_normal_(tensor, mean=0.0, std=1.0):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)

class CMlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CBlock(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#        self.attn = nn.Conv2d(dim, dim, 13, padding=6, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x + self.drop_path(self.conv2(self.attn(mask * self.conv1(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))))
        else:
            x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))))
        x = x + self.drop_path(self.mlp(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))
        return x

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0]
        ), f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert (
            W == self.img_size[1]
        ), f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.proj(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.act(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if rel_pos_bias is not None:
           attn = attn + rel_pos_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, rel_pos_bias=None):
        x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


class ConvViT(nn.Module):
    """Vision Transformer.

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        representation_size=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        init_std=0.02,
        init_values=None,
        beit_qkv_bias=False,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        img_size = [1024, 256, 128]
        embed_dim=[256, 384, 768]
        patch_size=[4, 2, 2]
        depth=[2, 2, 11]
        mlp_ratio=[4, 4, 4]
        num_heads=12
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_features = (
            self.embed_dim
        ) = embed_dim[-1]  # num_features for consistency with other models
        self.num_tokens = None
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed1 = embed_layer(
            img_size=img_size[0],
            patch_size=patch_size[0],
            in_chans=in_chans,
            embed_dim=embed_dim[0],
        )
        self.patch_embed2 = embed_layer(
            img_size=img_size[1],
            patch_size=patch_size[1],
            in_chans=embed_dim[0],
            embed_dim=embed_dim[1],
        )
        self.patch_embed3 = embed_layer(
            img_size=img_size[2],
            patch_size=patch_size[2],
            in_chans=embed_dim[1],
            embed_dim=embed_dim[2],
        )
        self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])
        num_patches = self.patch_embed3.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim[-1])
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))
        ]  # stochastic depth decay rule
        qk_scale = None
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads, mlp_ratio=mlp_ratio[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads, mlp_ratio=mlp_ratio[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i + depth[0]], norm_layer=norm_layer)
            for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([
            Block(
                dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratio[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[depth[0] + depth[1] + i], norm_layer=norm_layer)
            for i in range(depth[2])])

        self.norm = norm_layer(embed_dim[-1])

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("fc", nn.Linear(embed_dim, representation_size)),
                        ("act", nn.Tanh()),
                    ]
                )
            )
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.init_std = init_std
        trunc_normal_(self.pos_embed, std=self.init_std)
        self.apply(self._init_weights)
#        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            lecun_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(
            x.shape[0], -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return self.pre_logits(x[:, 0])

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class ConvViTDet(ConvViT, Backbone):
    def __init__(
        self,
        window_size,
        with_cp=False,
        pretrained="",
        stop_grad_conv1=False,
        sincos_pos_embed=False,
        zero_pos_embed=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # remove unusable parameters
        self.cls_token = None
        self.head = None
        self.norm = None

        unexpected_keys, missing_keys = [], []
        logger = setup_logger(name=__name__)
        if pretrained:  # only support load from local file
            checkpoint = torch.load(pretrained, map_location="cpu")

            for k in self.state_dict().keys():
                if k not in checkpoint["model"].keys():
                    missing_keys.append(k)
            for k in checkpoint["model"].keys():
                if k not in self.state_dict().keys():
                    unexpected_keys.append(k)

            if "pos_embed" in checkpoint["model"]:
                checkpoint["model"]["pos_embed"] = resize_pos_embed(
                    checkpoint["model"]["pos_embed"],
                    self.pos_embed,
                    self.num_tokens,
                    self.patch_embed3.grid_size,
                )
            self.load_state_dict(checkpoint["model"], strict=False)
            logger.info(f"Loading ViT pretrained weights from {pretrained}.")
            logger.warn(f"missing keys: {missing_keys}")
            logger.warn(f"unexpected keys: {unexpected_keys}")
        else:
            logger.info("Loading ViT pretrained weights from scratch.")
        self.window_size = window_size
        self.with_cp = with_cp
        self.ms_adaptor = nn.ModuleList(
            [
                nn.Identity(),
                nn.Identity(),
                nn.Identity(),
                nn.MaxPool2d(2),
            ]
        )
        self.ms_adaptor.apply(self.init_adaptor)

        self.windowed_rel_pos_bias = RelativePositionBias(
            window_size=(self.window_size, self.window_size), num_heads=self.num_heads
        )
        self.global_rel_pos_bias = RelativePositionBias(
            window_size=self.patch_embed3.grid_size, num_heads=self.num_heads
        )
        if "rel_pos_bias.relative_position_bias_table" in unexpected_keys:
            windowed_relative_position_bias_table = resize_pos_embed(
                checkpoint["model"]["rel_pos_bias.relative_position_bias_table"][
                    None, :-3
                ],
                self.windowed_rel_pos_bias.relative_position_bias_table[None],
                0,
            )
            global_relative_position_bias_table = resize_pos_embed(
                checkpoint["model"]["rel_pos_bias.relative_position_bias_table"][
                    None, :-3
                ],
                self.global_rel_pos_bias.relative_position_bias_table[None],
                0,
            )
            self.windowed_rel_pos_bias.load_state_dict(
                {
                    "relative_position_bias_table": windowed_relative_position_bias_table[
                        0
                    ]
                },
                strict=False,
            )
            self.global_rel_pos_bias.load_state_dict(
                {
                    "relative_position_bias_table": global_relative_position_bias_table[
                        0
                    ]
                },
                strict=False,
            )
            logger.info("Load positional bias table from checkpoint.")

        self._out_features = ["s0", "s1", "s2", "s3"]
        self._out_feature_channels = {"s0": 256, "s1": 384, "s2": 768, "s3": 768}
        self._out_feature_strides = {"s0": 4, "s1": 8, "s2": 16, "s3": 32}
        self.global_attn_index = [0, 4, 8, 10]
        # stop grad conv1
        self.stop_grad_conv1 = stop_grad_conv1
        if stop_grad_conv1:
            self.patch_embed.proj.weight.requires_grad = False
            self.patch_embed.proj.bias.requires_grad = False

        # sincos pos embed
        self.sincos_pos_embed = sincos_pos_embed
        if sincos_pos_embed:
            self.build_2d_sincos_position_embedding()
        if zero_pos_embed:
            nn.init.constant_(self.pos_embed, 0.0)

#        # remove pos_embed for extra tokens
#        self.pos_embed = nn.Parameter(self.pos_embed[:, self.num_tokens :, :])

    def init_adaptor(self, m):
        if isinstance(m, nn.Conv2d):
            lecun_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.ConvTranspose2d):
            lecun_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def build_2d_sincos_position_embedding(self, temperature=10000.0):
        h, w = self.patch_embed3.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert (
            self.embed_dim % 4 == 0
        ), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature ** omega)
        out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
        out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
        pos_emb = torch.cat(
            [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)],
            dim=1,
        )[None, :, :]
        self.pos_embed.requires_grad = False

    def forward(self, images):
        outs = dict()
        x = images.tensor
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        outs['s0'] = x
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        outs['s1'] = x
        x = self.patch_embed3(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.patch_embed4(x)
        x = x + self.pos_embed

        x = self.blocks3[0](x, self.global_rel_pos_bias())
        x = rearrange(
                x,
                "b (h w) c -> b h w c",
                h=self.patch_embed3.grid_size[0],
                w=self.patch_embed3.grid_size[1],
        )
        x = rearrange(
                x,
                "b (h h1) (w w1) c -> (b h w) (h1 w1) c",
                h1=self.window_size,
                w1=self.window_size,
        )
        for blk in self.blocks3[1: 3]:    
            x = blk(x, self.windowed_rel_pos_bias()) 
        
        x = rearrange(
                x,
                "(b h w) (h1 w1) c -> b (h h1 w w1) c",
                h=self.patch_embed3.grid_size[0] // self.window_size,
                w=self.patch_embed3.grid_size[1] // self.window_size,
                h1=self.window_size,
                w1=self.window_size,
        ) 
        x = self.blocks3[3](x, self.global_rel_pos_bias())
        x = rearrange(
                x,
                "b (h w) c -> b h w c",
                h=self.patch_embed3.grid_size[0],
                w=self.patch_embed3.grid_size[1],
        )
        x = rearrange(
                x,
                "b (h h1) (w w1) c -> (b h w) (h1 w1) c",
                h1=self.window_size,
                w1=self.window_size,
        )
        for blk in self.blocks3[4: 6]:
            x = blk(x, self.windowed_rel_pos_bias())

        x = rearrange(
                x,
                "(b h w) (h1 w1) c -> b (h h1 w w1) c",
                h=self.patch_embed3.grid_size[0] // self.window_size,
                w=self.patch_embed3.grid_size[1] // self.window_size,
                h1=self.window_size,
                w1=self.window_size,
        )
        x = self.blocks3[6](x, self.global_rel_pos_bias())
        x = rearrange(
                x,
                "b (h w) c -> b h w c",
                h=self.patch_embed3.grid_size[0],
                w=self.patch_embed3.grid_size[1],
        )
        x = rearrange(
                x,
                "b (h h1) (w w1) c -> (b h w) (h1 w1) c",
                h1=self.window_size,
                w1=self.window_size,
        )
        for blk in self.blocks3[7: 10]:
            x = blk(x, self.windowed_rel_pos_bias())

        x = rearrange(
                x,
                "(b h w) (h1 w1) c -> b (h h1 w w1) c",
                h=self.patch_embed3.grid_size[0] // self.window_size,
                w=self.patch_embed3.grid_size[1] // self.window_size,
                h1=self.window_size,
                w1=self.window_size,
        )
        x = self.blocks3[10](x, self.global_rel_pos_bias())
        x = rearrange(
                x,
                "b (h w) c -> b c h w",
                h=self.patch_embed3.grid_size[0],
                w=self.patch_embed3.grid_size[1],
        )
        outs["s2"] = x
        outs["s3"] = self.ms_adaptor[-1](x)



        return outs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger = logging.getLogger(__name__)
    _logger.info("Resized position embedding: %s to %s", posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info("Position embedding grid-size from %s to %s", [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(
        posemb_grid, size=gs_new, mode="bicubic", align_corners=False
    )
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb
