#%%
import sys
sys.path.insert(0, "/home/junkataoka/AVATAR")
import torch
# content of test_sample.py
#%%
vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
#%%
inp = torch.rand((2, 3, 384,384))


# %%
out = vits16(inp)
print(out.shape)
print(inp.shape)
