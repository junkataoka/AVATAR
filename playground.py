#%%
import torch
a = torch.load("temp_index.pt", map_location=torch.device("cpu"))
b = torch.load("temp_src_cs.pt", map_location=torch.device("cpu"))

print(b[a])