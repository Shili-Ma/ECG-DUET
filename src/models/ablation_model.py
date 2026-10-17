from dataclasses import asdict, dataclass
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ecg_model import DynamicFusion, FourierBlock, MoE, SpatialBackbone, count_params


@dataclass(frozen=True)
class AblationConfig:
    name: str
    use_time_branch: bool = True
    use_spatial_branch: bool = True
    use_idcnn: bool = True
    use_dynamic_fusion: bool = True
    use_moe: bool = True
    use_fourier_block: bool = True

    def model_kwargs(self) -> dict:
        values = asdict(self)
        values.pop("name")
        return values


DEFAULT_ABLATIONS: List[AblationConfig] = [
    AblationConfig(
        name="1_Full_Model",
        use_time_branch=True,
        use_spatial_branch=True,
        use_idcnn=True,
        use_dynamic_fusion=True,
        use_moe=True,
        use_fourier_block=True,
    ),
    AblationConfig(
        name="2_No_Time_Branch",
        use_time_branch=False,
        use_spatial_branch=True,
        use_idcnn=True,
        use_dynamic_fusion=False,
        use_moe=True,
        use_fourier_block=False,
    ),
    AblationConfig(
        name="3_No_Spatial_Branch",
        use_time_branch=True,
        use_spatial_branch=False,
        use_idcnn=False,
        use_dynamic_fusion=False,
        use_moe=True,
        use_fourier_block=True,
    ),
    AblationConfig(
        name="4_No_IDCNN",
        use_time_branch=True,
        use_spatial_branch=True,
        use_idcnn=False,
        use_dynamic_fusion=True,
        use_moe=True,
        use_fourier_block=True,
    ),
    AblationConfig(
        name="5_No_Dynamic_Fusion",
        use_time_branch=True,
        use_spatial_branch=True,
        use_idcnn=True,
        use_dynamic_fusion=False,
        use_moe=True,
        use_fourier_block=True,
    ),
    AblationConfig(
        name="6_No_MoE",
        use_time_branch=True,
        use_spatial_branch=True,
        use_idcnn=True,
        use_dynamic_fusion=True,
        use_moe=False,
        use_fourier_block=True,
    ),
    AblationConfig(
        name="7_No_Fourier_Block",
        use_time_branch=True,
        use_spatial_branch=True,
        use_idcnn=True,
        use_dynamic_fusion=True,
        use_moe=True,
        use_fourier_block=False,
    ),
]


def get_ablation_configs(selected: Iterable[str] | None = None) -> List[AblationConfig]:
    if selected is None:
        return list(DEFAULT_ABLATIONS)

    selected_names = list(selected)
    if len(selected_names) == 1 and selected_names[0].lower() == "all":
        return list(DEFAULT_ABLATIONS)

    by_name = {config.name: config for config in DEFAULT_ABLATIONS}
    missing = [name for name in selected_names if name not in by_name]
    if missing:
        available = ", ".join(by_name.keys())
        raise ValueError(f"Unknown ablation config(s): {missing}. Available configs: {available}")
    return [by_name[name] for name in selected_names]


class ColumnPatchToSequence(nn.Module):
    """
    Convert an ECG image to a width-wise sequence by learning a column encoder.
    This preserves the rectangular ECG layout used by the original ablation code.
    """

    def __init__(self, img_height: int, img_width: int, embed_dim: int = 128):
        super().__init__()
        self.img_height = img_height
        self.seq_len = img_width
        self.column_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=embed_dim // 2, kernel_size=(img_height, 1)),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(in_channels=embed_dim // 2, out_channels=embed_dim, kernel_size=1),
        )
        self.pos_emb = nn.Parameter(torch.randn(1, embed_dim, self.seq_len) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        if gray.shape[-2] != self.img_height or gray.shape[-1] != self.seq_len:
            gray = F.interpolate(
                gray,
                size=(self.img_height, self.seq_len),
                mode="bilinear",
                align_corners=False,
            )
        h = self.column_encoder(gray).squeeze(2)
        h = h + self.pos_emb
        return h.transpose(1, 2)


class SwitchableTimeTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        use_fourier_block: bool = True,
    ):
        super().__init__()
        self.use_fourier_block = use_fourier_block
        if self.use_fourier_block:
            self.fourier = FourierBlock(d_model, keep_ratio=0.25)
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
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fourier_block:
            x = self.fourier(x)
        x = self.encoder(x)
        x = x.transpose(1, 2)
        return self.pool(x).squeeze(-1)


class AblationModel(nn.Module):
    def __init__(
        self,
        img_height: int,
        img_width: int = 1064,
        t_embed: int = 128,
        t_layers: int = 2,
        s_idcnn: int = 128,
        fusion_out: int = 256,
        num_classes: int = 3,
        pretrained_backbone: bool = False,
        use_time_branch: bool = True,
        use_spatial_branch: bool = True,
        use_idcnn: bool = True,
        use_dynamic_fusion: bool = True,
        use_moe: bool = True,
        use_fourier_block: bool = True,
    ):
        super().__init__()
        if not use_time_branch and not use_spatial_branch:
            raise ValueError("At least one branch must be enabled for an ablation model.")

        self.use_time_branch = use_time_branch
        self.use_spatial_branch = use_spatial_branch
        self.use_dynamic_fusion = use_dynamic_fusion and use_time_branch and use_spatial_branch
        self.use_moe = use_moe

        if self.use_time_branch:
            self.p2s = ColumnPatchToSequence(img_height=img_height, img_width=img_width, embed_dim=t_embed)
            self.time_enc = SwitchableTimeTransformer(
                d_model=t_embed,
                nhead=4,
                num_layers=t_layers,
                dim_feedforward=4 * t_embed,
                use_fourier_block=use_fourier_block,
            )
            self.t_dim = t_embed
        else:
            self.p2s = None
            self.time_enc = None
            self.t_dim = 0

        if self.use_spatial_branch:
            self.spatial = SpatialBackbone(
                idcnn_out=s_idcnn,
                pretrained=pretrained_backbone,
                use_idcnn=use_idcnn,
            )
            self.s_dim = self.spatial.out_dim
        else:
            self.spatial = None
            self.s_dim = 0

        fusion_in_dim = self.t_dim + self.s_dim
        if self.use_dynamic_fusion:
            self.fusion = DynamicFusion(t_dim=self.t_dim, s_dim=self.s_dim, out_dim=fusion_out)
        else:
            self.fusion = nn.Linear(fusion_in_dim, fusion_out)

        if self.use_moe:
            self.moe_or_mlp = MoE(in_dim=fusion_out, hidden=256, num_experts=4, k=2)
        else:
            self.moe_or_mlp = nn.Sequential(
                nn.Linear(fusion_out, 256),
                nn.GELU(),
                nn.Linear(256, 256),
                nn.GELU(),
            )
        self.head = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_time_branch:
            t_feat = self.time_enc(self.p2s(x))
        else:
            t_feat = torch.empty(x.size(0), 0, device=x.device, dtype=x.dtype)

        if self.use_spatial_branch:
            s_feat = self.spatial(x)
        else:
            s_feat = torch.empty(x.size(0), 0, device=x.device, dtype=x.dtype)

        if self.use_dynamic_fusion:
            z = self.fusion(t_feat, s_feat)
        else:
            z = self.fusion(torch.cat([t_feat, s_feat], dim=1))
        z = self.moe_or_mlp(z)
        return self.head(z)


__all__ = [
    "AblationConfig",
    "AblationModel",
    "ColumnPatchToSequence",
    "DEFAULT_ABLATIONS",
    "SwitchableTimeTransformer",
    "count_params",
    "get_ablation_configs",
]
