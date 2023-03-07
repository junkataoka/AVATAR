import torch
from models.vision_transformer import VisionTransformer

input = torch.randn(1, 3, 255, 255)
print(input.shape)
print("0:", input.shape)
model = VisionTransformer()
output = model(input)
print("3:", output.shape)
