#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import glob
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size

#%%
a2w_path = "/home/junkataoka/AVATAR/checkpoints/office31_adapt_amazon2webcam_bs32_resnet50_lr0.0001_domain-adv_dis-src_dis-tar_dis-feat-src_dis-feat-tar_conf-pseudo-label_tsne"
acc_csv = pd.read_csv(os.path.join(a2w_path, "acc.csv"))
th_csv = pd.read_csv(os.path.join(a2w_path, "df_th.csv"))
mu_csv = pd.read_csv(os.path.join(a2w_path, "df_mu.csv"))
sd_csv = pd.read_csv(os.path.join(a2w_path, "df_sd.csv"))


# %%
acc = acc_csv.groupby("epoch")["test_acc"].max().values
th = th_csv.mean(axis=1).values
mu = mu_csv.mean(axis=1).values
sd = sd_csv.mean(axis=1).values

# %%
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(acc, "g-", label="Accuracy")
ax2.plot(th, "b-", label=r"$\tau$")
ax1.set_xlabel("Epoch")
ax1.set_ylabel(r"Accuracy")
ax2.set_ylabel(r"$\tau$")
ax1.legend(loc=3)
ax2.legend(loc=4)
fig.suptitle(r'Average across 31 classes', fontsize=16)
plt.savefig("figures/tau/average_tau.png")
plt.show()
plt.clf()
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
epochsize = 200
for i in range(epochsize):
   tsne2_path = f"/home/junkataoka/AVATAR/tsne/office31_adapt_amazon2webcam_bs32_resnet50_lr0.001_domain-adv_dis-src_dis-tar_dis-feat-src_dis-feat-tar_conf-pseudo-label_tsne_tsne_df2_epoch{i}.csv"

   tsne_path = f"/home/junkataoka/AVATAR/tsne/office31_adapt_amazon2webcam_bs32_resnet50_lr0.001_domain-adv_dis-src_dis-tar_dis-feat-src_dis-feat-tar_conf-pseudo-label_tsne_tsne_df_epoch{i}.csv"


   label_path = f"/home/junkataoka/AVATAR/tsne/office31_adapt_amazon2webcam_bs32_resnet50_lr0.001_domain-adv_dis-src_dis-tar_dis-feat-src_dis-feat-tar_conf-pseudo-label_tsne_label_df_epoch{i}.csv"


   df_tsne = pd.read_csv(tsne_path, index_col=0)
   df_tsne2 = pd.read_csv(tsne2_path, index_col=0)
   df_label = pd.read_csv(label_path, index_col=0)
   df_tsne2 = pd.concat([df_tsne2, df_label], axis=1)
   df_tsne = pd.concat([df_tsne, df_label], axis=1)

   plt.tick_params(left = False, right = False , labelleft = False,
                  labelbottom = False, bottom = False)
   sns.scatterplot(x="0", y="1", hue="Domain label", data=df_tsne, legend=False, size=5)
   plt.title(f"Epoch {i+1}", size=10)
   plt.savefig(f"figures/tsne/z0/tsne_epoch{i+1}", bbox_inches='tight')
   plt.clf()
   plt.tick_params(left = False, right = False , labelleft = False,
                  labelbottom = False, bottom = False)
   sns.scatterplot(x="0", y="1", hue="Domain label", data=df_tsne2, legend=False, size=5)
   plt.title(f"Epoch {i+1}", size=10)
   plt.savefig(f"figures/tsne/z1/tsne2_epoch{i+1}", bbox_inches='tight')
   plt.clf()

# %%

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