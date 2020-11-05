# ------------------------------------------------------------------------------
# This code is base on 
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        """
        Initialize k layer.

        Args:
            self: (todo): write your description
            k: (int): write your description
            inp_dim: (int): write your description
            out_dim: (int): write your description
            stride: (int): write your description
            with_bn: (str): write your description
        """
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Perform algorithm.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

class fully_connected(nn.Module):
    def __init__(self, inp_dim, out_dim, with_bn=True):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            inp_dim: (int): write your description
            out_dim: (int): write your description
            with_bn: (str): write your description
        """
        super(fully_connected, self).__init__()
        self.with_bn = with_bn

        self.linear = nn.Linear(inp_dim, out_dim)
        if self.with_bn:
            self.bn = nn.BatchNorm1d(out_dim)
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward function.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        linear = self.linear(x)
        bn     = self.bn(linear) if self.with_bn else linear
        relu   = self.relu(bn)
        return relu

class residual(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        """
        Initialize k layer.

        Args:
            self: (todo): write your description
            k: (int): write your description
            inp_dim: (int): write your description
            out_dim: (int): write your description
            stride: (int): write your description
            with_bn: (str): write your description
        """
        super(residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1   = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2   = nn.BatchNorm2d(out_dim)
        
        self.skip  = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Perform forward forward algorithm.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2   = self.bn2(conv2)

        skip  = self.skip(x)
        return self.relu(bn2 + skip)

def make_layer(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    """
    Make a layer.

    Args:
        k: (todo): write your description
        inp_dim: (int): write your description
        out_dim: (int): write your description
        modules: (list): write your description
        layer: (todo): write your description
        convolution: (bool): write your description
    """
    layers = [layer(k, inp_dim, out_dim, **kwargs)]
    for _ in range(1, modules):
        layers.append(layer(k, out_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)

def make_layer_revr(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    """
    Make a layer layer layers.

    Args:
        k: (todo): write your description
        inp_dim: (int): write your description
        out_dim: (int): write your description
        modules: (list): write your description
        layer: (todo): write your description
        convolution: (bool): write your description
    """
    layers = []
    for _ in range(modules - 1):
        layers.append(layer(k, inp_dim, inp_dim, **kwargs))
    layers.append(layer(k, inp_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)

class MergeUp(nn.Module):
    def forward(self, up1, up2):
        """
        Perform of two - up to1 and up.

        Args:
            self: (todo): write your description
            up1: (todo): write your description
            up2: (todo): write your description
        """
        return up1 + up2

def make_merge_layer(dim):
    """
    Make a merge layer.

    Args:
        dim: (int): write your description
    """
    return MergeUp()

# def make_pool_layer(dim):
#     return nn.MaxPool2d(kernel_size=2, stride=2)

def make_pool_layer(dim):
    """
    Make a layer layer.

    Args:
        dim: (int): write your description
    """
    return nn.Sequential()

def make_unpool_layer(dim):
    """
    Make a layer layer. layer.

    Args:
        dim: (int): write your description
    """
    return nn.Upsample(scale_factor=2)

def make_kp_layer(cnv_dim, curr_dim, out_dim):
    """
    Make a convolution layer.

    Args:
        cnv_dim: (int): write your description
        curr_dim: (int): write your description
        out_dim: (int): write your description
    """
    return nn.Sequential(
        convolution(3, cnv_dim, curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, (1, 1))
    )

def make_inter_layer(dim):
    """
    Make a layer.

    Args:
        dim: (int): write your description
    """
    return residual(3, dim, dim)

def make_cnv_layer(inp_dim, out_dim):
    """
    Make a convolution layer.

    Args:
        inp_dim: (int): write your description
        out_dim: (int): write your description
    """
    return convolution(3, inp_dim, out_dim)

class kp_module(nn.Module):
    def __init__(
        self, n, dims, modules, layer=residual,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, **kwargs
    ):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            n: (int): write your description
            dims: (str): write your description
            modules: (str): write your description
            layer: (todo): write your description
            residual: (todo): write your description
            make_up_layer: (todo): write your description
            make_layer: (todo): write your description
            make_low_layer: (todo): write your description
            make_layer: (todo): write your description
            make_hg_layer: (todo): write your description
            make_layer: (todo): write your description
            make_hg_layer_revr: (todo): write your description
            make_layer_revr: (todo): write your description
            make_pool_layer: (todo): write your description
            make_pool_layer: (todo): write your description
            make_unpool_layer: (todo): write your description
            make_unpool_layer: (todo): write your description
            make_merge_layer: (todo): write your description
            make_merge_layer: (todo): write your description
        """
        super(kp_module, self).__init__()

        self.n   = n

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.up1  = make_up_layer(
            3, curr_dim, curr_dim, curr_mod, 
            layer=layer, **kwargs
        )  
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(
            3, curr_dim, next_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.low2 = kp_module(
            n - 1, dims[1:], modules[1:], layer=layer, 
            make_up_layer=make_up_layer, 
            make_low_layer=make_low_layer,
            make_hg_layer=make_hg_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_pool_layer=make_pool_layer,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer,
            **kwargs
        ) if self.n > 1 else \
        make_low_layer(
            3, next_dim, next_dim, next_mod,
            layer=layer, **kwargs
        )
        self.low3 = make_hg_layer_revr(
            3, next_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.up2  = make_unpool_layer(curr_dim)

        self.merge = make_merge_layer(curr_dim)

    def forward(self, x):
        """
        Returns the forward of two ). )

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        up1  = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return self.merge(up1, up2)

class exkp(nn.Module):
    def __init__(
        self, n, nstack, dims, modules, heads, pre=None, cnv_dim=256, 
        make_tl_layer=None, make_br_layer=None,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer, 
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer, 
        kp_layer=residual
    ):
        """
        Initialize layer.

        Args:
            self: (todo): write your description
            n: (int): write your description
            nstack: (todo): write your description
            dims: (str): write your description
            modules: (str): write your description
            heads: (todo): write your description
            pre: (todo): write your description
            cnv_dim: (int): write your description
            make_tl_layer: (todo): write your description
            make_br_layer: (todo): write your description
            make_cnv_layer: (todo): write your description
            make_cnv_layer: (todo): write your description
            make_heat_layer: (todo): write your description
            make_kp_layer: (todo): write your description
            make_tag_layer: (todo): write your description
            make_kp_layer: (todo): write your description
            make_regr_layer: (todo): write your description
            make_kp_layer: (todo): write your description
            make_up_layer: (todo): write your description
            make_layer: (todo): write your description
            make_low_layer: (todo): write your description
            make_layer: (todo): write your description
            make_hg_layer: (todo): write your description
            make_layer: (todo): write your description
            make_hg_layer_revr: (todo): write your description
            make_layer_revr: (todo): write your description
            make_pool_layer: (todo): write your description
            make_pool_layer: (todo): write your description
            make_unpool_layer: (todo): write your description
            make_unpool_layer: (todo): write your description
            make_merge_layer: (todo): write your description
            make_merge_layer: (todo): write your description
            make_inter_layer: (int): write your description
            make_inter_layer: (int): write your description
            kp_layer: (todo): write your description
            residual: (todo): write your description
        """
        super(exkp, self).__init__()

        self.nstack    = nstack
        self.heads     = heads

        curr_dim = dims[0]

        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre

        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        ## keypoint heatmaps
        for head in heads.keys():
            if 'hm' in head:
                module =  nn.ModuleList([
                    make_heat_layer(
                        cnv_dim, curr_dim, heads[head]) for _ in range(nstack)
                ])
                self.__setattr__(head, module)
                for heat in self.__getattr__(head):
                    heat[-1].bias.data.fill_(-2.19)
            else:
                module = nn.ModuleList([
                    make_regr_layer(
                        cnv_dim, curr_dim, heads[head]) for _ in range(nstack)
                ])
                self.__setattr__(head, module)


        self.relu = nn.ReLU(inplace=True)

    def forward(self, image):
        """
        Perform forward

        Args:
            self: (todo): write your description
            image: (array): write your description
        """
        # print('image shape', image.shape)
        inter = self.pre(image)
        outs  = []

        for ind in range(self.nstack):
            kp_, cnv_  = self.kps[ind], self.cnvs[ind]
            kp  = kp_(inter)
            cnv = cnv_(kp)

            out = {}
            for head in self.heads:
                layer = self.__getattr__(head)[ind]
                y = layer(cnv)
                out[head] = y
            
            outs.append(out)
            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return outs


def make_hg_layer(kernel, dim0, dim1, mod, layer=convolution, **kwargs):
    """
    Make a convolution layer.

    Args:
        kernel: (str): write your description
        dim0: (int): write your description
        dim1: (int): write your description
        mod: (todo): write your description
        layer: (todo): write your description
        convolution: (bool): write your description
    """
    layers  = [layer(kernel, dim0, dim1, stride=2)]
    layers += [layer(kernel, dim1, dim1) for _ in range(mod - 1)]
    return nn.Sequential(*layers)


class HourglassNet(exkp):
    def __init__(self, heads, num_stacks=2):
        """
        Initialize layer.

        Args:
            self: (todo): write your description
            heads: (todo): write your description
            num_stacks: (int): write your description
        """
        n       = 5
        dims    = [256, 256, 384, 384, 384, 512]
        modules = [2, 2, 2, 2, 2, 4]

        super(HourglassNet, self).__init__(
            n, num_stacks, dims, modules, heads,
            make_tl_layer=None,
            make_br_layer=None,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            kp_layer=residual, cnv_dim=256
        )

def get_large_hourglass_net(num_layers, heads, head_conv):
    """
    Get a netglass model.

    Args:
        num_layers: (int): write your description
        heads: (todo): write your description
        head_conv: (todo): write your description
    """
  model = HourglassNet(heads, 2)
  return model
