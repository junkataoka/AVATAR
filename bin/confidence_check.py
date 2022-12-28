#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import glob
import matplotlib as mpl
import numpy as np
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size

#%%
tasks = ["amazon2dslr", "amazon2webcam", "webcam2amazon", "dslr2amazon", "webcam2dslr", "dslr2webcam"]
labels = [r"A$\rightarrow$D", r"A$\rightarrow$W", r"W$\rightarrow$A", r"D$\rightarrow$A", r"W$\rightarrow$D", r"D$\rightarrow$W"]

fig, ax1 = plt.subplots(figsize=(6, 4))

for idx, task in enumerate(tasks):
   path = f"~/AVATAR/checkpoints/office31_adapt_{task}_bs32_resnet50_lr0.001_domain-adv_dis-src_dis-tar_conf-pseudo-labelID1"
   acc_csv = pd.read_csv(os.path.join(path, "acc.csv"))

   acc = acc_csv.groupby("epoch")["test_acc"].max().values
   ax1.plot(acc, "-", label=f"{labels[idx]}")

ax1.set_xlabel("Epoch", fontsize=16)
ax1.set_ylabel("Accuracy", fontsize=16)
ax1.yaxis.set_ticks(np.arange(50, 105, 5))

ax1.legend(loc=4)
plt.savefig(f"~/figures/accuracy.png", bbox_inches="tight", dpi=300)

#%%

fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.set_ylim(0.7, 1)
for idx, task in enumerate(tasks):
   path = f"~/AVATAR/checkpoints/office31_adapt_{task}_bs32_resnet50_lr0.001_domain-adv_dis-src_dis-tar_conf-pseudo-labelID1"

   th_csv = pd.read_csv(os.path.join(path, "df_th.csv"))
   # mu_csv = pd.read_csv(os.path.join(path, "df_mu.csv"))
   # sd_csv = pd.read_csv(os.path.join(path, "df_sd.csv"))

   th = th_csv.mean(axis=1).values
   # mu = mu_csv.mean(axis=1).values
   # sd = sd_csv.mean(axis=1).values

   ax1.plot(th, "-", label=f"{labels[idx]}")

ax1.set_xlabel("Epoch", fontsize=16)
ax1.yaxis.set_ticks(np.arange(0.7, 1, 0.05))

ax1.set_ylabel(r"Threshold", fontsize=16)

ax1.legend(loc=4)
plt.savefig(f"~/figures/threshold.png", bbox_inches="tight", dpi=300)
#%%
# ax1.set_title(r'Average $\tau$ across 31 classes', fontsize=16)
# ax2.set_title(r'Average accuracy across 31 classes', fontsize=16)

# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.plot(acc, "g-", label="Accuracy")
# ax2.plot(th, "b-", label=r"$\tau$")
# ax1.set_xlabel("Epoch")
# ax2.set_ylabel(r"$\tau$")
# ax1.legend(loc=3)
# ax2.legend(loc=4)
# fig.suptitle(r'Average across 31 classes', fontsize=16)
# plt.show()
# plt.savefig("figures/tau/average_tau.png")
# plt.clf()
# %%
fig, ax = plt.subplots(nrows=6, ncols=6, figsize=(100, 80))
c = 0
for i in range(6):
   for j in range(6):
      if c < 31:
         acc = acc_csv.groupby("epoch")[f"test_acc_class_{c+1}"].max().values
         th = th_csv.iloc[:, c].values
         ax2 = ax[i, j].twinx()
         ax[i,j].plot(acc, "g-", label="Accuracy")
         ax2.plot(th, "b-", label=r"$\tau$")
         ax[i,j].set_xlabel("Epoch", fontsize=20)
         ax[i,j].set_ylabel(r"Accuracy", fontsize=20)
         ax[i,j].set_title(f'class {c+1}', fontsize=20)
         ax[i,j].legend(loc=3, fontsize=20)
         ax2.legend(loc=4, fontsize=20)
         c += 1

plt.savefig(f"figures/tau/classwise_tau.png", bbox_inches="tight")
plt.clf()

# %%
i = 20
# tsne2_path = f"/home/junkataoka/AVATAR/tsne/office31_adapt_amazon2webcam_bs32_resnet50_lr0.001_domain-adv_dis-src_dis-tar_dis-feat-src_dis-feat-tar_conf-pseudo-label_tsne_tsne_df2_epoch{i}.csv"

tsne2_path = "/data/home/jkataok1/AVATAR2022/tsne/office31_adapt_amazon2webcam_bs32_resnet50_lr0.001_domain-adv_dis-src_dis-tar_conf-pseudo-labelID1_tsne_df2_epoch0.csv"

label_path = "/data/home/jkataok1/AVATAR2022/tsne/office31_adapt_amazon2webcam_bs32_resnet50_lr0.001_domain-adv_dis-src_dis-tar_conf-pseudo-labelID1_label_df_epoch0.csv"

tsne_path = "/data/home/jkataok1/AVATAR2022/tsne/office31_adapt_amazon2webcam_bs32_resnet50_lr0.001_domain-adv_dis-src_dis-tar_conf-pseudo-labelID1_tsne_df_epoch0.csv"


df_tsne = pd.read_csv(tsne_path, index_col=0)
df_tsne2 = pd.read_csv(tsne2_path, index_col=0)
df_label = pd.read_csv(label_path, index_col=0)

df_tsne2 = pd.concat([df_tsne2, df_label], axis=1)
df_tsne = pd.concat([df_tsne, df_label], axis=1)

# plt.scatter(df_tsne2.iloc[:, 0], df_tsne2.iloc[:, 1], s=5)
#%%
# df_tsne
# sns.scatterplot(x="0", y="1", hue="True_label", data=df_tsne2, legend=False, size=5)

plt.tick_params(left = False, right = False , labelleft = False,
               labelbottom = False, bottom = False)

df_tsne2["True_label"] = df_tsne2["True_label"].astype("str")
sns.scatterplot(x="0", y="1", hue="Domain label", data=df_tsne2, legend=False, s=15, ec=None)
plt.savefig("tsne.png", dpi=300, bbox_inches="tight")
# plt.savefig(f"figures/tsne/z0/tsne_epoch{i+1}", bbox_inches='tight')
#%%
import plotly.express as px
df_tsne2["size"] = 0.1
fig = px.scatter(df_tsne2, x="0", y="1", color="Domain label", hover_data=["Path"])
fig.show()

# %%
df_tsne2

# %%
images = glob.glob('figures/tsne/z0/*.png')
images.sort(key=lambda x: int(x.split("epoch")[-1].split(".")[0]))
frame = cv2.imread(images[0])
height, width, layers = frame.shape

video = cv2.VideoWriter('figures/tsne/videos/z0_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, (width, height))

for filename in images:
    img = cv2.imread(filename)
    video.write(img)

video.release()
# %%
# %%
# %%

images = glob.glob('figures/tsne/z1/*.png')
images.sort(key=lambda x: int(x.split("epoch")[-1].split(".")[0]))
frame = cv2.imread(images[0])
height, width, layers = frame.shape

video = cv2.VideoWriter('figures/tsne/videos/z1_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, (width, height))

for filename in images:
    img = cv2.imread(filename)
    video.write(img)

video.release()
# %%
# %%
images