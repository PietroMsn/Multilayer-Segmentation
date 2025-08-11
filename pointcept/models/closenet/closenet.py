import torch
import torch.nn as nn
import torch.nn.init as init

import pointops
from pointcept.models.builder import MODELS


class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return (
                self.norm(input.transpose(1, 2).contiguous())
                .transpose(1, 2)
                .contiguous()
            )
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError


class EdgeConv(nn.Module):
    """
    input: (N, 6) + (N,)
    output: (N, 64) + (N,)

    1. get_graph_feature(x): (N, k, 6*2) (knn on features vectors)
    2. mlp: (N, k, 64)
    3. max pooling: (N, 64)
    """

    def __init__(self, channels: list[int], k: int):
        super().__init__()
        self.k = k
        convs = []
        in_ch = channels[0]
        for out_ch in channels[1:]:
            convs.extend(
                (
                    nn.Linear(in_ch, out_ch, bias=False),
                    PointBatchNorm(out_ch),
                    nn.LeakyReLU(negative_slope=0.2),
                )
            )
            in_ch = out_ch
        self.s_mlp = nn.Sequential(*convs)

    def forward(self, pxo):
        p, x, o = pxo

        idx, _ = pointops.knn_query(self.k, x, o, None, None)  # (n, k), int32
        grouped_feats = pointops.grouping(
            idx, x, p, with_xyz=False
        )  # (n, k, 6), float32
        exp_feats = x.unsqueeze(1).expand(-1, self.k, -1)  # (n, k, 6), float32
        x_graph = torch.cat(
            (grouped_feats - exp_feats, exp_feats), dim=-1
        )  # (n, k, 12), float32

        out = self.s_mlp(x_graph)  # (n, k, outch)
        out = out.max(dim=1, keepdim=False)[0]  # (n, outch)

        return out


class TransformNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.mlp1 = nn.Sequential(
            nn.Linear(6, 64, bias=False),
            PointBatchNorm(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 128, bias=False),
            PointBatchNorm(128),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(128, 1024, bias=False),
            PointBatchNorm(1024),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.transform = nn.Linear(256, 3 * 3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x, offsets):
        """
        x: (n, k, 3+3)
        offsets: (b,)
        returns: (b, 3, 3)
        """
        out = self.mlp1(x)  # (n, k, 128)
        out = out.max(dim=1, keepdim=False)[0]  # (n, 128)

        out = self.mlp2(out)  # (n, 1024)
        outs = []
        # batch-wise max pool
        for i in range(offsets.shape[0]):
            if i == 0:
                s_i, e_i = 0, offsets[0]
            else:
                s_i, e_i = offsets[i - 1], offsets[i]
            out_b = out[s_i:e_i, :].max(dim=0, keepdim=False)[0]  # (1024,)
            outs.append(out_b)
        out = torch.stack(outs, 0)  # (b, 1024)

        out = self.transform(self.mlp3(out))
        out = out.view(offsets.shape[0], 3, 3)  # (b, 3, 3)

        return out


class DGCNNEnc(nn.Module):
    def __init__(self, input_dim, emb_dim, k, use_tnet):
        super().__init__()
        self.k = k
        self.tnet = TransformNet() if use_tnet else None
        self.conv1 = EdgeConv(channels=[input_dim * 2, 64, 64], k=k)
        self.conv2 = EdgeConv(channels=[64 * 2, 64, 64], k=k)
        self.conv3 = EdgeConv(channels=[64 * 2, 64], k=k)
        self.lin_global = nn.Sequential(
            nn.Linear(64 * 3, emb_dim, bias=False),
            PointBatchNorm(emb_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, pxo):
        """
        input: pxo as usual
        output: [(n, 64), (n, 64), (n, 64)], (n, 1024)
        """
        p, x, o = pxo

        if self.tnet:
            idx, _ = pointops.knn_query(self.k, p, o, None, None)  # (n, k), int32
            grouped_xyz = pointops.grouping2(p, idx)  # (n, k, 3), float32
            exp_xyz = p.unsqueeze(1).expand(-1, self.k, -1)  # (n, k, 3), float32
            xyz_graph = torch.cat(
                (grouped_xyz - exp_xyz, exp_xyz), dim=-1
            )  # (n, k, 3+3), float32
            t = self.tnet(xyz_graph, o)
            p_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i = 0, o[0]
                else:
                    s_i, e_i = o[i - 1], o[i]
                p_b = torch.matmul(p[s_i:e_i, :], t[i])
                p_tmp.append(p_b)
            p = torch.cat(p_tmp, 0)
            x = torch.cat([p, x[:, 3:]], -1)

        x1 = self.conv1([p, x, o]).float()
        x2 = self.conv2([p, x1, o]).float()
        x3 = self.conv3([p, x2, o]).float()

        x = torch.cat([x1, x2, x3], dim=1)  # (n, 64*3)
        global_enc = self.lin_global(x)  # (n, 1024)

        # batch-wise max pool
        outs = []
        for i in range(o.shape[0]):
            if i == 0:
                s_i, e_i = 0, o[0]
            else:
                s_i, e_i = o[i - 1], o[i]
            out_b = global_enc[s_i:e_i, :].max(dim=0, keepdim=False)[0]  # (1024,)
            outs.append(out_b)
        x_glob = torch.stack(outs, 0)  # (b, 1024)

        outs = []
        for i in range(o.shape[0]):
            if i == 0:
                sz = o[i]
            else:
                sz = o[i] - o[i - 1]
            outs.append(x_glob[i].unsqueeze(0).expand(sz, -1))
        globenc = torch.cat(outs, dim=0)

        return (x1, x2, x3), globenc


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
            self.segm_dec = nn.Sequential(
                nn.Linear(64 * 3 + pts_emb_dim, 512),
                PointBatchNorm(512),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(dropout),
                nn.Linear(512, 512),
                PointBatchNorm(512),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(dropout),
                nn.Linear(512, 512),
                PointBatchNorm(512),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                PointBatchNorm(256),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(dropout),
                nn.Linear(256, n_classes),
            )

    def forward(self, data_dict):
        p = data_dict["coord"]
        x = data_dict["feat"]
        o = data_dict["offset"].int()

        x = torch.cat((p, x), dim=-1)
        pxo = [p, x, o]

        # encode
        feats, fglob = self.pts_enc(pxo)  # 3*(n, 64) & (n, embdim)
        encodings = torch.cat([*feats, fglob], dim=-1)  # (n, 3*64 + embdim)

        # decode
        if self.use_seg_head:
            logits = self.segm_dec(encodings)
            return logits

        else:
            return encodings  # (n, 3*64 + embdim)
