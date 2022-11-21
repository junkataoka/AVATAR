#%%
import os
import pandas as pd
import matplotlib.pyplot as plt

#%%
a2w_path = "/data/home/jkataok1/AVATAR2022/checkpoints/office31_adapt_amazon2webcam_bs32_resnet50_lr0.001_domain-adv_dis-src_dis-tar_dis-feat-src_dis-feat-tar_conf-pseudo-label"
acc_csv = pd.read_csv(os.path.join(w2a_path, "acc.csv"))
th_csv = pd.read_csv(os.path.join(w2a_path, "df_th.csv"))
mu_csv = pd.read_csv(os.path.join(w2a_path, "df_mu.csv"))
sd_csv = pd.read_csv(os.path.join(w2a_path, "df_sd.csv"))


# %%
acc_csv

# %%
plt.plot(sd_csv.mean(axis=1), mu_csv.mean(axis=1))

# %%

for i in range(31):
   plt.plot(th_csv.iloc[:, i])
   plt.show()
# %%
