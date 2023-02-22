from models.resnet import *
from models.dino import *


def construct(args):
    if args.arch.find('resnet') != -1:
        if args.arch == 'resnet50':
            return resnet50(args)
        elif args.arch == 'resnet101':
            return resnet101(args)
    if args.arch.find('vit') != -1:
        if args.arch == "vits16":
            return dino_vits16(args)
        elif args.arch == "vits8":
            return dino_vits8(args)
        elif args.arch == "vitb16":
            return dino_vitb16(args)
        elif args.arch == "vitb8":
            return dino_vitb8(args)
    else:
        raise NotImplementedError
        