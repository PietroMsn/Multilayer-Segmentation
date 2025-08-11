"""
CloseNet without garments and body encoder: just dcgnn (?).

Might be a bit different from the original paper

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import torch.nn.init as init

from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn.models import MLP
from torch_geometric.nn import pool

from pointops import offset2batch
from pointcept.models.builder import MODELS


class TransformNetOld(nn.Module):
    def __init__(self, k):
        super().__init__()
        act_fun = "leaky_relu"
        act_args = {"negative_slope": 0.2}  # Weird, default=0.01
        self.conv1 = DynamicEdgeConv(
            nn=MLP(
                [3 * 2, 64, 128],
                bias=False,
                plain_last=False,
                act=act_fun,
                act_kwargs=act_args,
            ),
            k=k,
            aggr="max",
        )
        self.conv2 = DynamicEdgeConv(
            nn=MLP(
                [128 * 2, 1024],
                bias=False,
                plain_last=False,
                act=act_fun,
                act_kwargs=act_args,
            ),
            k=k,
            aggr="max",
        )

        self.mlp = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.Linear(256, 3 * 3),
        )

        init.constant_(self.mlp[-1].weight, 0)
        init.eye_(self.mlp[-1].bias.view(3, 3))

    def forward(self, pxo):
        p, x, o = pxo
        b = offset2batch(o)
        x = self.conv1(p, b)
        x = self.conv2(x, b)
        x = pool.global_max_pool(x, b, o.shape[0])
        return self.mlp(x).view(-1, 3, 3)


class DGCNNEncOld(nn.Module):
    def __init__(self, input_dim, emb_dim, k, use_tnet):
        super().__init__()
        act_fun = "leaky_relu"
        act_args = {"negative_slope": 0.2}  # Weird, default=0.01
        aggr = "max"

        self.tnet = TransformNetOld(k) if use_tnet else None
        self.conv1 = DynamicEdgeConv(
            MLP(
                [2 * input_dim, 64, 64],
                bias=False,
                plain_last=False,
                act=act_fun,
                act_kwargs=act_args,
            ),
            k,
            aggr,
        )
        self.conv2 = DynamicEdgeConv(
            MLP(
                [2 * 64, 64, 64],
                bias=False,
                plain_last=False,
                act=act_fun,
                act_kwargs=act_args,
            ),
            k,
            aggr,
        )
        self.conv3 = DynamicEdgeConv(
            MLP(
                [2 * 64, 64],
                bias=False,
                plain_last=False,
                act=act_fun,
                act_kwargs=act_args,
            ),
            k,
            aggr,
        )

        self.mlp = MLP(
            [3 * 64, emb_dim],
            bias=False,
            plain_last=False,
            act=act_fun,
            act_kwargs=act_args,
        )
        # mlp2 = MLP([  16, 64],   bias=False, plain_last=False, act=act_fun, act_kwargs=act_args)
        # drop_list=[0.6, 0.6, 0, 0]
        # self.mlp3 = MLP([1024 + 3*64, 256, 256, 128, part_num], plain_last=True, act=act_fun, act_kwargs=act_args, dropout=drop_list)

    def forward(self, pxo):
        p, x, o = pxo
        b = offset2batch(o)

        if self.tnet:
            t = self.tnet([p, x, o])
            p_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i = 0, o[0]
                else:
                    s_i, e_i = o[i - 1], o[i]
                p_b = torch.matmul(p[s_i:e_i, :], t[i])
                p_tmp.append(p_b)
            p = torch.cat(p_tmp, 0)
            x = torch.cat([p, x], -1)

        x1 = self.conv1(x, b)
        x2 = self.conv2(x1, b)
        x3 = self.conv3(x2, b)
        x_glob = self.mlp(torch.cat([x1, x2, x3], dim=1))
        x_glob = pool.global_max_pool(x_glob, b)

        outs = []
        for i in range(o.shape[0]):
            if i == 0:
                sz = o[i]
            else:
                sz = o[i] - o[i - 1]
            outs.append(x_glob[i].unsqueeze(0).expand(sz, -1))
        globenc = torch.cat(outs, dim=0)

        return (x1, x2, x3), globenc


"""
@MODELS.register_module()
class CloSeNet(nn.Module):
    def __init__(
        self,
        inp_dim,
        pts_emb_dim,
        k,
        use_tnet,
        n_classes,
        dropout,
        use_seg_head=True,
    ):
        super().__init__()
        self.pts_enc = DGCNNEnc(inp_dim, pts_emb_dim, k, use_tnet)

        self.use_seg_head = use_seg_head
        if self.use_seg_head:
        self.segm_dec = MLP(
            channel_list=[
                64 * 3 + pts_emb_dim,
            ]
            + [
                512,
            ]
            * 4
            + [256, n_classes],
            dropout=dropout,
            act="leaky_relu",
            act_kwargs=dict(negative_slope=0.2),
        )

    def forward(self, data_dict):
        p = data_dict["coord"]
        x = data_dict["feat"]
        o = data_dict["offset"].int()
        pxo = [p, x, o]

        # encode
        feats, fglob = self.pts_enc(pxo)  # 3*(n, 64) & (n, embdim)
        encodings = torch.cat([*feats, fglob], dim=-1)

        # decode
        if self.use_seg_head:
            logits = self.segm_dec(encodings)
            return logits

        else:
            return encodings  # (n, 3*64 + embdim)
"""
