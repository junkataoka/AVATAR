#%%
import torch
import math
import numpy as np
import matplotlib.pyplot as plt

#%%
max_epoch = 200
epoch = np.arange(200)
#%%
lam = [2 / (1.0 + math.exp(-1.0 * 10 * epoch[i] / max_epoch)) - 1 for i in range(200)]
# %%
plt.plot(lam)

# %%

lam
# %%
