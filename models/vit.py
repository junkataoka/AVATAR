import torch
import torch.nn as nn

class MyViT():
    def __init__(self, n_class, image_size) -> None:
        super().__init__()
        self.vit = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        self.num_of_eature_map = image_size
        self.fc1 = nn.Sequential(nn.Linear(image_size, image_size), 
                                nn.BatchNorm1d(image_size),
                                nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(image_size, n_class)

    def forward(self, x):
        x = self.vit(x)
        x2 = self.fc1(x)
        ca = self.fc2(x)

        return x, x2, ca
