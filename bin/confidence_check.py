%%
import os
import pandas as pd
import matplotlib.pyplot as plt

#%%
a2w_path = "/home/junkataoka/AVATAR/checkpoints/office31_adapt_amazon2webcam_bs32_resnet50_lr0.001_domain-adv_dis-src_dis-tar_dis-feat-src_dis-feat-tar_conf-pseudo-label"
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
ax2.plot(th, "b-", label=r"Average of $\tau$")
ax1.set_xlabel("Epoch")
ax1.set_ylabel(r"Accuracy")
ax2.set_ylabel(r"Average of $\tau$")
ax1.legend(loc=3)
ax2.legend(loc=4)
# %%
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(acc, "g-", label="Accuracy")
ax2.plot(mu, "b-", label=r"Average of $\mu$")
ax1.set_xlabel("Epoch")
ax1.set_ylabel(r"Accuracy")
ax2.set_ylabel(r"Average of $\mu$")
ax1.legend(loc=3)
ax2.legend(loc=4)
# %%
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(acc, "g-", label="Accuracy")
ax2.plot(sd, "b-", label=r"Average of $\sigma$")
ax1.set_xlabel("Epoch")
ax1.set_ylabel(r"Accuracy")
ax2.set_ylabel(r"Average of $\sigma$")
ax1.legend(loc=3)
ax2.legend(loc=4)
# %%
acc = acc_csv.groupby("epoch")["test_acc_class_6"].max().values
th = th_csv.iloc[:, 5].values
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(acc, "g-", label="Accuracy for class 6")
ax2.plot(th, "b-", label=r"$\tau$ for class 6")
ax1.set_xlabel("Epoch")
ax1.set_ylabel(r"Accuracy for class 6")
ax2.set_ylabel(r"$\tau$ for class 6")
ax1.legend(loc=3)
ax2.legend(loc=4)

# %%
acc = acc_csv.groupby("epoch")["test_acc_class_26"].max().values
th = th_csv.iloc[:, 25].values
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(acc, "g-", label="Accuracy for class 26")
ax2.plot(th, "b-", label=r"$\tau$ for class 26")
ax1.set_xlabel("Epoch")
ax1.set_ylabel(r"Accuracy for class 26")
ax2.set_ylabel(r"$\tau$ for class 26")
ax1.legend(loc=3)
ax2.legend(loc=4)

# %%
acc = acc_csv.groupby("epoch")["test_acc_class_26"].max().values
mu = mu_csv.iloc[:, 25].values
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(acc, "g-", label="Accuracy for class 26")
ax2.plot(mu, "b-", label=r"$\mu$ for class 26")
ax1.set_xlabel("Epoch")
ax1.set_ylabel(r"Accuracy for class 26")
ax2.set_ylabel(r"$\mu$ for class 26")
ax1.legend(loc=3)
ax2.legend(loc=4)

#%%
acc = acc_csv.groupby("epoch")["test_acc_class_26"].max().values
sd = sd_csv.iloc[:, 25].values
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(acc, "g-", label="Accuracy for class 26")
ax2.plot(sd, "b-", label=r"$\sigma$ for class 26")
ax1.set_xlabel("Epoch")
ax1.set_ylabel(r"Accuracy for class 26")
ax2.set_ylabel(r"$\sigma$ for class 26")
ax1.legend(loc=3)
ax2.legend(loc=4)

# %%
