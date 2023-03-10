##%
import sys
sys.path.insert(0, "/home/junkataoka/AVATAR")
import torch
import argparse
from models.model_construct import construct
from models.dino import *
# content of test_sample.py

# Implement MyViTs16 test case with DINO model

def test_vit():
    args = argparse.Namespace()
    args.arch = 'vitb16'
    args.pretrained_path = None
    args.num_classes = 10
    model = construct(args)
    model = model
    inp = torch.randn((2, 3, 224, 224))
    with torch.no_grad():
        feat, pred = model(inp)

# Check if weights are loaded correctly to the model with DINO model
def test_DINO():
    args = argparse.Namespace()
    args.arch = 'vitb16'
    args.pretrained_path = None
    args.num_classes = 10
    model = construct(args)
    model = model
    weight_sum = torch.sum(list(model.parameters())[0])

if __name__ == "__main__":
    test_vit()
    test_DINO()

