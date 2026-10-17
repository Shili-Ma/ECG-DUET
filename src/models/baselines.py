import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResizeInputModel(nn.Module):
    def __init__(self, model: nn.Module, size: int | tuple[int, int]):
        super().__init__()
        self.model = model
        self.size = size

    def forward(self, x):
        x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
        return self.model(x)


class PatchToSequence(nn.Module):
    def __init__(self, img_height: int, img_width: int, embed_dim: int = 128):
        super().__init__()
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
        h = self.column_encoder(gray).squeeze(2)
        h = h + self.pos_emb
        return h.transpose(1, 2)


class SequenceModel(nn.Module):
    def __init__(self, img_height: int, img_width: int, num_classes: int, model_type: str = "lstm"):
        super().__init__()
        embed_dim = 128
        hidden_dim = 256
        self.p2s = PatchToSequence(img_height=img_height, img_width=img_width, embed_dim=embed_dim)
        if model_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=0.2,
            )
        elif model_type.lower() == "gru":
            self.rnn = nn.GRU(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=0.2,
            )
        else:
            raise ValueError("model_type must be 'lstm' or 'gru'")
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.p2s(x)
        x, _ = self.rnn(x)
        return self.fc(x[:, -1, :])


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def se_resnet18(num_classes: int, pretrained: bool = False):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, layer_name)
        for block in layer:
            block.add_module("se_block", SEBlock(block.conv2.out_channels))
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_baseline_model(
    model_name: str,
    num_classes: int,
    img_height: int,
    img_width: int,
    pretrained: bool = False,
):
    name = model_name.lower()
    if name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif name == "se_ecgnet":
        model = se_resnet18(num_classes, pretrained)
    elif name == "vit":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        model = ResizeInputModel(model, (224, 224))
    elif name == "swintransformer":
        model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
        model.head = nn.Linear(model.head.in_features, num_classes)
        model = ResizeInputModel(model, (224, 224))
    elif name == "lstm":
        model = SequenceModel(img_height, img_width, num_classes, model_type="lstm")
    elif name == "gru":
        model = SequenceModel(img_height, img_width, num_classes, model_type="gru")
    else:
        raise ValueError(f"Unknown baseline model: {model_name}")
    return model


DEFAULT_BASELINES = [
    "resnet18",
    "densenet121",
    "mobilenet_v2",
    "efficientnet_b0",
    "se_ecgnet",
    "vit",
    "swintransformer",
    "lstm",
    "gru",
]

