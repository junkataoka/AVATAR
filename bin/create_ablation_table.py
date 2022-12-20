#%%
import os
import pandas as pd
from itertools import permutations
import numpy as np
#%%
def helper_find_best_acc(text):
   """_summary_
   Find  best accuracy from a text containing...
      'Test on T test set - epoch: 14, loss: 2.166530, Top1 acc: 54.753723, Top5 acc: 75.945017'

   Args:
       text (.txt): .txt to search for the best acc
   """
   out = []
   with open(text, "r") as f:
      lines = f.readlines()
      for l in lines:
         if l.find("Test on T test set") != -1:
            out.append(np.round(float(l.split("Top1 acc: ")[-1].split(",")[0]),1))

   return max(out)
# %%
# domains = ["art", "clipart", "realworld", "product"]
domains = ["amazon", "webcam", "dslr"]
domain_combinations = list(permutations(domains, 2))
methods = ["domain-adv", "dis-src", "dis-tar", "dis-feat-src", "dis-feat-tar", "conf-pseudo-label"]
out = pd.DataFrame(columns = ["2".join(domain_combinations[i]) for i in range(len(domain_combinations))])
row_idx = []

for j in range(1, len(methods)+1):
   row_name = "_".join(methods[:j])
   row_idx.append(row_name)
row_idx += ["domain-adv_dis-src_dis-tar_conf-pseudo-label"]
print(row_idx)
#%%
out['row'] = row_idx
out.set_index("row", inplace=True, drop=True)


for i in range(len(domain_combinations)):
   colname = "2".join(domain_combinations[i])
   for j in range(len(row_idx)):
      path = f'/data/home/jkataok1/AVATAR2022/checkpoints/office31_adapt_{colname}_bs32_resnet50_lr0.001_{row_idx[j]}/log.txt'
      acc = helper_find_best_acc(path)
      out.loc[row_idx[j], colname] = acc

#out.to_csv("office_home_ablation.csv")

out["Avg"] = out.mean(axis=1)
out
#%%
out[["art2clipart", "art2product", "art2realworld", "clipart2art", "clipart2product", "clipart2realworld", "product2art",
     "product2clipart", "product2realworld", "realworld2art", "realworld2clipart", "realworld2product", "Avg"]]
#%%

# %%
md = out.to_markdown()
with open("ablation_table.txt", "w") as f:
   f.write(md)
   f.close()





# %%
len(md)

# %%

row_name
# %%
