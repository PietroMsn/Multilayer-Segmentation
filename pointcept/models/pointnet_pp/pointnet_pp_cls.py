"""
PointNet++ for Object Classification

Author: Me.
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import pointops

from pointcept.models.builder import MODELS


class PointNetSetAbstraction(nn.Module):
    """
    Similar to `TransitionDown`, but with a PointNet block attached
    instead of a simple linear layer.

    With respect to PointNet++ paper, it performs:

    1. sampling
    2. grouping
    3. pointnet++ block

    Note that (1) and (2) are already done in `TransitionDown`.
    """

    def __init__(self, sampling_sz, nsamples, input_dim, mlp, group_all):
        """
        - sampling_sz: how many centroids per pointcloud when sampling
        - nsamples: n. of points to pick per group when grouping
        - mlp: n. of features per convolution in mlp when applying pointnet
        - group_all: wether to use the previous parameters or just building
                     one centroid per point cloud and grouping all remaining points.
        """
        super().__init__()
        self.group_all = group_all
        self.sampling_sz = sampling_sz
        self.nsamples = nsamples
        self.bns = nn.ModuleList()
        self.lins = nn.ModuleList()
        for output_dim in mlp:
            self.lins.append(nn.Linear(input_dim, output_dim, bias=False))
            self.bns.append(nn.BatchNorm1d(output_dim))
            input_dim = output_dim
        self.relu = nn.ReLU(inplace=True)

    def apply_mlp(self, x, batched=True):
        if batched:
            for lin, bn in zip(self.lins, self.bns):
                x = lin(x).transpose(1, 2).contiguous()
                x = bn(x).transpose(1, 2).contiguous()
                x = self.relu(x)
        else:
            for lin, bn in zip(self.lins, self.bns):
                x = self.relu(bn(lin(x)))
        return x

    def forward(self, pxo):
        """
        Input: a list of [pts (n, 3), feats (n, c), offsets (b)]
        Output: new_pts (n', 3), new_feats (n', out_dim), new_offsets (b)
        """
        p, x, o = pxo  # (n, 3), (n, c), (b)

        if self.group_all:
            # 1. sampling and grouping to one group only.
            # One centroid per each group, mapped to zero vector.
            """
            # Note: it is mandatory that the batches have same dimensions,
            #       since pointnet cannot be applied to batched input.
            # TODO: check batch dimensions
            b = o.shape[0]
            _, c = p.shape
            _, d = x.shape
            dev = p.device
            n_p = torch.zeros(
                b, c, dtype=torch.float, device=dev
            )  # centroids are (0, 0, 0)
            n_o = torch.arange(b, device=dev) + 1
            x = torch.cat(
                [x.reshape(b, -1, d), p.reshape(b, -1, c)], dim=2
            )  # (b, nsamples, c+d)
            # are you sure that n_x[0, :] are p2[:o2[0]] and x2[:o2[0]]?
            # yes... kind of (I trust reshaping)
            """

            x = torch.cat([x, p], dim=1)
            x = self.apply_mlp(x, batched=False)
            n_x = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
                x_b = x[s_i:e_i, :].max(0, True)[0]
                n_x.append(x_b)
            n_x = torch.cat(n_x, 0)
            n_p = torch.zeros(n_x.shape[0], 3, dtype=p.dtype, device=p.device)
            n_o = torch.arange(o.shape[0], device=o.device) + 1

        else:
            # 1. sampling: get centroids locations
            n_o = torch.tensor(  # new offsets
                [self.sampling_sz * (i + 1) for i in range(o.shape[0])], device=o.device
            )
            idx = pointops.farthest_point_sampling(p, o, n_o)
            n_p = p[idx.long(), :]  # new coordinates (n', 3)

            # 2. grouping: group nsamples per each centroid
            x, _ = pointops.knn_query_and_group(
                x,  # original features (n, c)
                p,  # original coordinates (n, 3)
                offset=o,  # original offsets (b)
                new_xyz=n_p,  # centroids coordinates (m, 3)
                new_offset=n_o,  # n. centroids per batch (b)
                nsample=self.nsamples,  # n. samples per centroid
                with_xyz=True,  # include xyz in features
            )  # (n', nsamples, c+3)

            # 3. apply pointnet++ and max pool to sampled groups
            x = self.apply_mlp(x)  # (n', nsamples, outdim)
            n_x = torch.max(x, dim=1)[0]  # (n', outdim)

        return n_p, n_x, n_o


@MODELS.register_module()
class PointNetCls(nn.Module):
    def __init__(self, feat_dim, embed_dim):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(512, 32, feat_dim + 3, [64, 64, 128], False)
        self.sa2 = PointNetSetAbstraction(128, 64, 128 + 3, [128, 128, 256], False)
        self.sa3 = PointNetSetAbstraction(
            None, None, 256 + 3, [256, 512, embed_dim], True
        )

    def forward(self, data_dict):
        p0 = data_dict["coord"]
        x0 = data_dict["feat"]
        o0 = data_dict["offset"].int()
        p1, x1, o1 = self.sa1([p0, x0, o0])
        p2, x2, o2 = self.sa2([p1, x1, o1])
        p3, x3, o3 = self.sa3([p2, x2, o2])
        return x3
