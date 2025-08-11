"""
PointNet++ for Semantic Segmentation

Author: Me.
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import pointops

from pointcept.models.builder import MODELS
from .pointnet_pp_cls import PointNetSetAbstraction


class PointNetFeaturePropagation(nn.Module):
    """
    Similar to `TransitionUp` of point_transformer_seg.
    Given as input the two outputs, the first one from the skip connection
    and the second one from the incoming vector, it performs 3 things:

    1. interpolate incoming vector to match the skip connection dimension
    2. concatenate features between interpolated and skip connection
       (when skip features are available)
    3. run an mlp (pointnet++) on the features
    """

    def __init__(self, input_dim, mlp):
        super().__init__()
        self.mlp = nn.Sequential()
        for output_dim in mlp:
            self.mlp.extend(
                (
                    nn.Linear(input_dim, output_dim, bias=False),
                    nn.BatchNorm1d(output_dim),
                    nn.ReLU(inplace=True),
                )
            )
            input_dim = output_dim

    def forward(self, pxo1, pxo2):
        """
        Input:
        - pxo1 [(n, 3), (n, c), (n)]: data from skip connection (higher n. of pts)
        - pxo2 [(m, 3), (m, d), (m)]: data to be interpolated (usually from previous block)

        Note: pxo1 features can be None. In that case, features are not concatenated
        and the mlp is fed with only interpolated features from pxo2.

        Output:
        - n_x (n, output_dim): new features
        """
        p1, x1, o1 = pxo1
        p2, x2, o2 = pxo2

        # 1. interpolate previous + concatenate with skip
        x2_interp = pointops.interpolation(p2, p1, x2, o2, o1)  # (n, d)
        if x1 is None:
            x_n = x2_interp
        else:
            x_n = torch.cat([x1, x2_interp], dim=1)  # (n, d+c)

        # 2. run mlp/pointnet++
        # Note: it is mandatory that batches have same dimensions,
        #       otherwise batchnorm has no sense...
        # b, n = o1.shape[0], o1[0]  # batches, n.pts per batch
        # for conv, bn in zip(self.convs, self.bns):
        #     x_n = conv(x_n.transpose(0, 1)).transpose(0, 1)
        #     x_n = x_n.view(b, n, -1)
        #     x_n = bn(x_n.transpose(1, 2)).transpose(1, 2)
        #     x_n = x_n.reshape(-1, x_n.shape[-1])
        # return x_n.contiguous()  # TODO: Review all process of batch normalization...

        return self.mlp(x_n)


@MODELS.register_module()
class PointNetSeg(nn.Module):
    def __init__(self, feat_dim, num_classes=6, use_segm_head=True):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(1024, 32, feat_dim + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(512 + 256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(256 + 128, [256, 256])
        self.fp2 = PointNetFeaturePropagation(256 + 64, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        self.use_segm_head = use_segm_head
        if use_segm_head:
            self.seg_head = nn.Sequential(
                nn.Conv1d(128, 128, 1),
                nn.BatchNorm1d(128),
                nn.Dropout(0.5),
                nn.Conv1d(128, num_classes, 1),
            )

    def forward(self, data_dict):
        p0 = data_dict["coord"]
        x0 = data_dict["feat"]
        o0 = data_dict["offset"].int()

        p1, x1, o1 = self.sa1([p0, x0, o0])
        p2, x2, o2 = self.sa2([p1, x1, o1])
        p3, x3, o3 = self.sa3([p2, x2, o2])
        p4, x4, o4 = self.sa4([p3, x3, o3])

        x3_n = self.fp4([p3, x3, o3], [p4, x4, o4])
        x2_n = self.fp3([p2, x2, o2], [p3, x3_n, o3])
        x1_n = self.fp2([p1, x1, o1], [p2, x2_n, o2])
        x0_n = self.fp1([p0, None, o0], [p1, x1_n, o1])

        if self.use_segm_head:
            x0_n = x0_n.view(o0.shape[0], o0[0], -1)
            out = self.seg_head(x0_n.transpose(1, 2)).transpose(1, 2)
            out_unbatch = out.reshape(o0[-1], -1)
            return out_unbatch
        else:
            return x0_n  # (n, 128)
