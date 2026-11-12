"""Model definitions - must match the training notebook exactly (Block 8).
Weights come from your own checkpoints, so nothing is downloaded."""
import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 4


def disable_inplace(model):
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = False
    return model


def build_resnet18():
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, NUM_CLASSES))
    return m


def build_squeezenet():
    m = models.squeezenet1_1(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Conv2d(512, NUM_CLASSES, 1),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())
    return m


def build_efficientnet_b0():
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    return m


def build_vit_tiny():
    import timm
    return timm.create_model("vit_tiny_patch16_224", pretrained=False,
                             num_classes=NUM_CLASSES, drop_rate=0.1)


def build_deit_tiny():
    import timm
    return timm.create_model("deit_tiny_patch16_224", pretrained=False,
                             num_classes=NUM_CLASSES, drop_rate=0.1)


class ChannelAttentionGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        r = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, r, bias=False), nn.ReLU(),
                                nn.Linear(r, channels, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.shape
        return x * self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)


class AGCTNet(nn.Module):
    def __init__(self, num_classes=4, d_model=256, n_heads=8, n_layers=3, dropout=0.3,
                 use_cnn=True, use_deit=True, use_cag=True, use_transformer=True):
        super().__init__()
        import timm
        self.use_cnn, self.use_deit, self.use_cag, self.use_tf = use_cnn, use_deit, use_cag, use_transformer
        ch = 0
        if use_cnn:
            sq = models.squeezenet1_1(weights=None)
            self.cnn_branch = sq.features
            self.cnn_align = nn.AdaptiveAvgPool2d((7, 7)); ch += 512
        if use_deit:
            self.tr_branch = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=0)
            self.tr_spatial = nn.Sequential(nn.Linear(192, 192 * 49, bias=False), nn.GELU()); ch += 192
        self.fuse_proj = nn.Sequential(nn.Conv2d(ch, d_model * 2, 1, bias=False),
                                       nn.BatchNorm2d(d_model * 2), nn.GELU())
        if use_cag:
            self.cag = ChannelAttentionGate(d_model * 2, 16)
        if use_transformer:
            self.patch_proj = nn.Conv2d(d_model * 2, d_model, 1)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos_embed = nn.Parameter(torch.zeros(1, 50, d_model))
            enc = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout,
                                             batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(enc, n_layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d_model); feat = d_model
        else:
            self.pool = nn.AdaptiveAvgPool2d(1); feat = d_model * 2
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat, num_classes))

    def forward(self, x):
        B, fs = x.size(0), []
        if self.use_cnn:
            fs.append(self.cnn_align(self.cnn_branch(x)))
        if self.use_deit:
            fs.append(self.tr_spatial(self.tr_branch(x)).view(B, 192, 7, 7))
        fused = self.fuse_proj(torch.cat(fs, 1))
        if self.use_cag:
            fused = self.cag(fused)
        if not self.use_tf:
            return self.head(self.pool(fused).flatten(1))
        tok = self.patch_proj(fused).flatten(2).permute(0, 2, 1)
        tok = torch.cat([self.cls_token.expand(B, -1, -1), tok], 1) + self.pos_embed
        return self.head(self.norm(self.transformer(tok)[:, 0]))


BUILDERS = {"ResNet-18": build_resnet18, "SqueezeNet1.1": build_squeezenet,
            "EfficientNet-B0": build_efficientnet_b0, "ViT-Tiny": build_vit_tiny,
            "DeiT-Tiny": build_deit_tiny, "AGCTNet": lambda: AGCTNet(NUM_CLASSES)}