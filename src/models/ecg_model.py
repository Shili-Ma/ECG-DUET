import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class PatchToSequence(nn.Module):
    def __init__(self, img_size: int = 256, seq_len: int = None, embed_dim: int = 128, pool: str = "mean"):
        super().__init__()
        self.pool = pool
        self.embed_dim = embed_dim
        self.seq_len = seq_len or img_size
        self.conv = nn.Sequential(
            nn.Conv1d(1, embed_dim // 2, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=5, padding=2),
        )
        self.pos_emb = nn.Parameter(torch.randn(1, embed_dim, self.seq_len) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        gray = x.mean(dim=1)
        if self.pool == "mean":
            seq = gray.mean(dim=1)
        else:
            seq, _ = gray.max(dim=1)
        seq = seq.unsqueeze(1)
        if W != self.seq_len:
            seq = F.interpolate(seq, size=self.seq_len, mode="linear", align_corners=False)
        h = self.conv(seq) + self.pos_emb
        return h.transpose(1, 2)


class FourierBlock(nn.Module):
    def __init__(self, d_model: int, keep_ratio: float = 0.25):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        X = torch.fft.rfft(x, dim=1)
        T_fft = X.size(1)
        k = max(1, int(T_fft * self.keep_ratio))
        X_low = torch.zeros_like(X)
        X_low[:, :k, :] = X[:, :k, :]
        x_low = torch.fft.irfft(X_low, n=x.size(1), dim=1)
        return x + self.gate(x_low)


class TimeTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fourier = FourierBlock(d_model, keep_ratio=0.25)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fourier(x)
        x = self.encoder(x)
        x = x.transpose(1, 2)
        return self.pool(x).squeeze(-1)


class IDCNNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilations=None):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8]
        layers = []
        _in = in_ch
        for d in dilations:
            layers += [
                nn.Conv2d(_in, out_ch, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            ]
            _in = out_ch
        self.net = nn.Sequential(*layers)
        self.proj = nn.Conv2d(out_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.proj(self.net(x))


class SpatialBackbone(nn.Module):
    def __init__(self, out_dim: int = 256, idcnn_out: int = 128, pretrained: bool = False, use_idcnn: bool = True):
        super().__init__()
        self.use_idcnn = use_idcnn
        base = resnet18(weights=None if not pretrained else "IMAGENET1K_V1")
        base.fc = nn.Identity()
        self.cnn = base
        if self.use_idcnn:
            self.idcnn = IDCNNBlock(in_ch=64, out_ch=idcnn_out)
            self.idcnn_out = idcnn_out
        else:
            self.idcnn = None
            self.idcnn_out = 0
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 512 + self.idcnn_out

    def forward(self, x):
        x0 = self.cnn.conv1(x)
        x0 = self.cnn.bn1(x0)
        x0 = self.cnn.relu(x0)
        x1 = self.cnn.maxpool(x0)
        x1 = self.cnn.layer1(x1)
        if self.use_idcnn:
            idc = self.idcnn(x1)
            l = self.pool(idc).flatten(1)
        else:
            l = torch.empty(x.size(0), 0, device=x.device)
        x2 = self.cnn.layer2(x1)
        x3 = self.cnn.layer3(x2)
        x4 = self.cnn.layer4(x3)
        g = self.pool(x4).flatten(1)
        return torch.cat([g, l], dim=1)


class DynamicFusion(nn.Module):
    def __init__(self, t_dim: int, s_dim: int, out_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(t_dim + s_dim, (t_dim + s_dim) // 2),
            nn.GELU(),
            nn.Linear((t_dim + s_dim) // 2, 2),
        )
        self.proj = nn.Linear(t_dim + s_dim, out_dim)

    def forward(self, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = torch.cat([t, s], dim=1)
        w = F.softmax(self.gate(x), dim=-1)
        t_weighted = t * w[:, 0:1]
        s_weighted = s * w[:, 1:2]
        z = torch.cat([t_weighted, s_weighted], dim=1)
        return self.proj(z)


class Expert(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class MoE(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, num_experts: int = 4, k: int = 2):
        super().__init__()
        self.k = k
        self.experts = nn.ModuleList([Expert(in_dim, hidden) for _ in range(num_experts)])
        self.gate = nn.Linear(in_dim, num_experts)
        self.out = nn.Linear(hidden, hidden)

    def forward(self, x):
        logits = self.gate(x)
        topk = torch.topk(logits, k=self.k, dim=-1)
        idx = topk.indices
        w = F.softmax(topk.values, dim=-1)
        out = 0.0
        for i in range(self.k):
            e_i = [self.experts[j] for j in idx[:, i].tolist()]
            parts = []
            for b in range(x.size(0)):
                parts.append(e_i[b](x[b:b+1]))
            parts = torch.cat(parts, dim=0)
            out = out + parts * w[:, i:i+1]
        return self.out(out)


class ECGThreeClassModel(nn.Module):
    def __init__(self, img_size=256, t_embed=128, t_layers=2, s_idcnn=128, fusion_out=256,
                 num_classes=3, pretrained_backbone=False):
        super().__init__()
        self.p2s = PatchToSequence(img_size=img_size, embed_dim=t_embed)
        self.time_enc = TimeTransformer(d_model=t_embed, nhead=4, num_layers=t_layers, dim_feedforward=4*t_embed)
        self.spatial = SpatialBackbone(out_dim=512 + s_idcnn, idcnn_out=s_idcnn, pretrained=pretrained_backbone)
        s_dim = 512 + s_idcnn
        self.fusion = DynamicFusion(t_dim=t_embed, s_dim=s_dim, out_dim=fusion_out)
        self.moe = MoE(in_dim=fusion_out, hidden=256, num_experts=4, k=2)
        self.head = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, num_classes))

    def forward(self, x):
        t_seq = self.p2s(x)
        t_feat = self.time_enc(t_seq)
        s_feat = self.spatial(x)
        z = self.fusion(t_feat, s_feat)
        z = self.moe(z)
        return self.head(z)


__all__ = [
    "count_params",
    "PatchToSequence",
    "FourierBlock",
    "TimeTransformer",
    "IDCNNBlock",
    "SpatialBackbone",
    "DynamicFusion",
    "Expert",
    "MoE",
    "ECGThreeClassModel",
]

