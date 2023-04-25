# AVATAR2022
Code release for AdVersarial self-superVised domain Adaptation network for TArget domain (AVATAR)

> **Abstract:** *We propose an unsupervised domain adaption (UDA) method to predict unlabeled data on the target domain, given labeled data from the source domain. Mainstream UDA models aim to learn domain-invariant features from the two domains or improve target discrimination based on labeled source domain data. However, such methods risk misclassifying the target domain when a discrepancy between the source and target domain is large. To tackle this problem, we propose an Adversarial self-superVised domain Adaptation network for TARget domain (AVATAR) algorithm, which outperforms state-of-the-art UDA models by reducing the domain discrepancy while enhancing the discrimination by a domain adversarial learning, deep clustering, and confidence-based pseudo-labeling strategy. Our proposed model significantly outperforms the state-of-the-art methods in two UDA benchmarks, and extensive ablation studies show the effectiveness of the proposed approach.*

# Table of Contents
  - [Architecture](#Architecture)
  - [Requirements](#Requirements)
  - [Dataset](#Dataset)
  - [Training](#Training)
  - [Performance](#Performance)
  - [Citation](#Citation)

# Architecture
![Model Architeuture](pics/avatar_architecture.png)

# Requirements
- Python 3.8.5
- Pytorch 1.9.1

# Dataset
## [Office-31](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view)
## [Office-home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?usp=sharing&resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw)

The structure of the dataset should be like
```
Office31
|_ amazon
|  |_ back_pack
|     |_ <im-1-name>.jpg
|     |_ ...
|  |_ bike
|     |_ <im-1-name>.jpg
|     |_ ...
|  |_ ... (omit 28 classes)
|  |_ trash_can
|     |_ <im-1-name>.jpg
|     |_ ...
|_ amazon_half
|  |_ back_pack
|     |_ <im-1-name>.jpg
|     |_ ...
|  |_ bike
|     |_ <im-1-name>.jpg
|     |_ ...
|  |_ ... (omit 28 classes)
|  |_ trash_can
|     |_ <im-1-name>.jpg
|     |_ ...
|_ amazon_half2
|  |_ back_pack
|     |_ <im-1-name>.jpg
|     |_ ...
|  |_ bike
|     |_ <im-1-name>.jpg
|     |_ ...
|  |_ ... (omit 28 classes)
|  |_ trash_can
|     |_ <im-1-name>.jpg
|     |_ ...
|_ dslr
|  |_ back_pack
|     |_ <im-1-name>.jpg
|     |_ ...
|  |_ bike
|     |_ <im-1-name>.jpg
|     |_ ...
|  |_ ... (omit 28 classes)
|  |_ trash_can
|     |_ <im-1-name>.jpg
|     |_ ...
|_ ...
```
# Training
Replace arguments in run_avatar.sh with those in your system.

# Performance
## Office 31
![Office-31 performance](pics/office31_avatar.jpg)
## Office-home
![Office-home performance](pics/office_home_avatar.jpg)
## Image-CLEF
![Image-CLEFT performance](pics/office_home_avatar.jpg)

# Citations
```
@InProceedings{AVATAR,
  title={AdVersarial self-superVised domain Adaptation network for TArget domain (AVATAR)},
  author={Jun Kataoka and Hyunsoo Yoon},
  year={2022},
}
```