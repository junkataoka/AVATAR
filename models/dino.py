import torch
import torch.nn as nn
from functools import partial
import models.vision_transformer as vits


dependencies = ["torch", "torchvision"]


class MyViTs16(vits.VisionTransformer):

    def __init__(self, n_class):
        super(MyViTs16, self).__init__(
            patch_size=16, embed_dim=384, depth=12,
            num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )

        self.fc1 = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(inplace=True)
            )
        self.fc2 = nn.Linear(self.embed_dim, n_class+1)

    def forward(self, x):

        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x[:, 0]
        x = torch.flatten(x, 1)
        x2 = self.fc1(x)
        ca = self.fc2(x2)
        return x2, ca
        

class MyViTs8(vits.VisionTransformer):

    def __init__(self, n_class):
        super(MyViTs8, self).__init__(
            patch_size=8, embed_dim=384, depth=12,
            num_heads=6, mlp_ratio=4,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.fc1 = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(self.embed_dim, n_class+1)

    def forward(self, x):

        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x[:, 0]
        x = torch.flatten(x, 1)
        x2 = self.fc1(x)
        ca = self.fc2(x2)
        return x2, ca


class MyViTb16(vits.VisionTransformer):

    def __init__(self, n_class):
        super(MyViTb16, self).__init__(
            patch_size=16, embed_dim=768, depth=12,
            num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.fc1 = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(self.embed_dim, n_class+1)

    def forward(self, x):

        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x[:, 0]
        x = torch.flatten(x, 1)
        x2 = self.fc1(x)
        ca = self.fc2(x2)
        return x2, ca


class MyViTb8(vits.VisionTransformer):

    def __init__(self, n_class):
        super(MyViTb8, self).__init__(
            patch_size=8, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.fc1 = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(inplace=True))

        self.fc2 = nn.Linear(self.embed_dim, n_class+1)

    def forward(self, x):

        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x[:, 0]
        x = torch.flatten(x, 1)
        x2 = self.fc1(x)
        ca = self.fc2(x2)
        return x2, ca

def dino_vits16(args, **kwargs):
    """
    ViT-Small/16x16 pre-trained with DINO.
    Achieves 74.5% top-1 accuracy on ImageNet with k-NN classification.
    """
    # model = vits.__dict__["vit_small"](patch_size=16, num_classes=0, **kwargs)
    model = MyViTs16(n_class=args.num_classes)

    if args.pretrained_path:
        model_dict = model.state_dict()
        pretrained_dict_temp = torch.load(args.pretrained_path)
        pretrained_dict = {k: v for k, v in pretrained_dict_temp.items() if k in model_dict}
        model.load_state_dict(pretrained_dict, strict=False)
        print(args.pretrained_path)
        print('Source pre-trained model has been loaded!')
    else:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=False)

    return model


def dino_vits8(args, **kwargs):
    """
    ViT-Small/8x8 pre-trained with DINO.
    Achieves 78.3% top-1 accuracy on ImageNet with k-NN classification.
    """
    model = MyViTs8(n_class=args.num_classes)

    if args.pretrained_path:
        model_dict = model.state_dict()
        pretrained_dict_temp = torch.load(args.pretrained_path)
        pretrained_dict = {k: v for k, v in pretrained_dict_temp.items() if k in model_dict}
        model.load_state_dict(pretrained_dict, strict=True)
        print(args.pretrained_path)
        print('Source pre-trained model has been loaded!')
    else:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=False)
    return model


def dino_vitb16(args, **kwargs):
    """
    ViT-Base/16x16 pre-trained with DINO.
    Achieves 76.1% top-1 accuracy on ImageNet with k-NN classification.
    """
    model = MyViTb16(n_class=args.num_classes)

    if args.pretrained_path:
        model_dict = model.state_dict()
        pretrained_dict_temp = torch.load(args.pretrained_path)
        pretrained_dict = {k: v for k, v in pretrained_dict_temp.items() if k in model_dict}
        model.load_state_dict(pretrained_dict, strict=False)
        print(args.pretrained_path)
        print('Source pre-trained model has been loaded!')
    else:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=False)
    return model


def dino_vitb8(args, **kwargs):
    """
    ViT-Base/8x8 pre-trained with DINO.
    Achieves 77.4% top-1 accuracy on ImageNet with k-NN classification.
    """
    model = MyViTb8(n_class=args.num_classes)

    if args.pretrained_path:
        model_dict = model.state_dict()
        pretrained_dict_temp = torch.load(args.pretrained_path)
        pretrained_dict = {k: v for k, v in pretrained_dict_temp.items() if k in model_dict}
        model.load_state_dict(pretrained_dict, strict=False)
        print(args.pretrained_path)
        print('Source pre-trained model has been loaded!')
    else:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=False)
    return model

