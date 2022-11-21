#%%
import os
import pandas as pd
from itertools import permutations
# %%
domains = ["art", "clipart", "realworld", "product"]
domain_combinations = list(permutations(domains, 2))
methods = ["domain-adv", "dis-src", "dis-tar", "dis-feat-src", "dis-feat-tar", "conf-pseudo-label"]

for i in range(len(domain_combinations)):
   for j in range(1, len(methods)+1):
      method = "_".join(methods[:j])
      path = f'/data/home/jkataok1/AVATAR2022/checkpoints/office31_adapt_{domain_combinations[i][0]}2{domain_combinations[i][1]}_bs32_resnet50_lr0.001_{method}/log.txt'
      with open(path) as f:
         lines = f.readlines()
         acc = lines[-5].split(" ")[6]
         print(acc)


# %%

