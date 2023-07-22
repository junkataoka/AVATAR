import torch.nn as nn
import torch
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck


class MyResNet50(ResNet):
    def __init__(self, n_class):
        super(MyResNet50, self).__init__(Bottleneck, [3, 4, 6, 3])
        num_of_feature_map = self.fc.in_features
        self.fc1 = nn.Sequential(nn.Linear(num_of_feature_map, num_of_feature_map//4),
        nn.BatchNorm1d(num_of_feature_map//4),
        nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(num_of_feature_map//4, n_class+1), nn.Softmax(dim=1))

    def forward(self, x):
        # change forward here
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x2 = self.fc1(x)
        ca = self.fc2(x2)
        return ca, x2

class MyResNet101(ResNet):
    def __init__(self, n_class):
        super(MyResNet101, self).__init__(Bottleneck, [3, 4, 23, 3])
        num_of_feature_map = self.fc.in_features
        self.fc1 = nn.Sequential(nn.Linear(num_of_feature_map, num_of_feature_map//4),
        nn.BatchNorm1d(num_of_feature_map//4),
        nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(num_of_feature_map//4, n_class+1)

    def forward(self, x):
        # change forward here
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x2 = self.fc1(x)
        ca = self.fc2(x2)
        return x2, ca


def resnet50(args, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MyResNet50(n_class=args.num_classes)

    if args.pretrained_path:
        model_dict = model.state_dict()
        pretrained_dict_temp = torch.load(args.pretrained_path)
        pretrained_dict = {k: v for k, v in pretrained_dict_temp.items()}

        for k1, k2 in zip(model_dict.keys(), pretrained_dict.keys()):
            model_dict[k1] = pretrained_dict[k2]

        model.load_state_dict(model_dict, strict=True)
        print(args.pretrained_path)
        print('Source pre-trained model has been loaded!')

    else:
        model.load_state_dict(models.resnet50(pretrained=True).state_dict(), strict=False)

    return model


def resnet101(args, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MyResNet101(n_class=args.num_classes)
    if args.pretrained_path:
        model_dict = model.state_dict()
        pretrained_dict_temp = torch.load(args.pretrained_path)
        pretrained_dict = {k: v for k, v in pretrained_dict_temp.items()}

        for k1, k2 in zip(model_dict.keys(), pretrained_dict.keys()):
            model_dict[k1] = pretrained_dict[k2]

        model.load_state_dict(model_dict, strict=True)
        print(args.pretrained_path)
        print('Source pre-trained model has been loaded!')

    else:
        model.load_state_dict(models.resnet101(pretrained=True).state_dict(), strict=False)

    return model


def construct(args, **kwargs):

    if args.arch == 'resnet50':
        return resnet50(args)
    elif args.arch == 'resnet101':
        return resnet101(args)
    else:
        raise ValueError('Unrecognized model architecture: ', args.arch)
