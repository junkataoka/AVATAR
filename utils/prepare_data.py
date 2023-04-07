import os
import torch
import torchvision.transforms as transforms
from utils.folder import ImageFolder
import numpy as np
import collections
import torch
from torch.utils.data import Dataset
import random

def createSubsetIndice(dataset, majorityClass, ratio):

    indice = [i for i in range(len(dataset))]
    labels = dataset.targets.numpy()
    indiceMap = collections.defaultdict(list)

    for l, i in zip(labels, indice):
        indiceMap[l].append(i)

    countMap = collections.Counter(labels)
    targetNumMino = int(round(float(countMap[majorityClass]) * ratio))

    for k in countMap.keys():
        if k != majorityClass:
            indiceMap[k] = np.random.choice(indiceMap[k], size=targetNumMino, replace=False)

    res = []
    for l in indiceMap.values():
        res.extend(l)
    return set(res)

class custom_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices, labels):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.targets = labels
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)


def subset_dataset(dataset, args):
    res = createSubsetIndice(dataset, args.majority_class, args.minority_class_ratio)
    selected_targets = [dataset.targets[i] for i in range(len(dataset.targets)) if i in res]
    new_dataset = custom_subset(dataset, res, selected_targets)
    return new_dataset


def generate_dataloader(args):
    # Data loading code
    traindir = os.path.join(args.data_path_source, args.src)
    traindir_t = os.path.join(args.data_path_target, args.tar)
    valdir = os.path.join(args.data_path_target, args.tar)
    valdir_t = os.path.join(args.data_path_target_t, args.tar_t)

    if not os.path.isdir(traindir):
        raise ValueError ('the require data path is not exist, please download the dataset')

        # transformation on the training data during training
    src_data_transform_train = transforms.Compose([
            transforms.Resize((256, 256)), # spatial size of vgg-f input
            transforms.RandomCrop((224,224)),
        transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tar_data_transform_train = transforms.Compose([
        transforms.Resize((256,256)),
            transforms.RandomCrop((224,224)),
        transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # transformation on the duplicated data during training
    data_transform_test = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    

    source_train_dataset = ImageFolder(root=traindir, transform=src_data_transform_train)
    source_test_dataset = ImageFolder(root=traindir, transform=data_transform_test)
    target_train_dataset = ImageFolder(root=traindir_t, transform=tar_data_transform_train)
    target_test_dataset = ImageFolder(root=valdir, transform=data_transform_test)
    target_test_dataset_t = ImageFolder(root=valdir_t, transform=tar_data_transform_train)

    if args.src_subset:
        random.seed(seed=10)
        source_train_dataset = subset_dataset(source_train_dataset, args)
        source_test_dataset = subset_dataset(source_test_dataset, args)
    if args.tar_subset:
        random.seed(seed=10)
        target_train_dataset = subset_dataset(target_train_dataset, args)
        target_test_dataset = subset_dataset(target_test_dataset, args)
        target_test_dataset_t = subset_dataset(target_test_dataset_t, args)

    source_train_loader = torch.utils.data.DataLoader(
        source_train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True
    )
    source_test_loader = torch.utils.data.DataLoader(
        source_test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    target_train_loader = torch.utils.data.DataLoader(
        target_train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True
    )
    target_test_loader = torch.utils.data.DataLoader(
        target_test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    target_test_loader_t = torch.utils.data.DataLoader(
        target_test_dataset_t, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    return source_train_loader, target_train_loader, target_test_loader, target_test_loader_t, source_test_loader



