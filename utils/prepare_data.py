import os
import torch
import torchvision.transforms as transforms
from utils.folder import ImageFolder
import numpy as np

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

    source_train_loader = torch.utils.data.DataLoader(
        source_train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True
    )
    source_test_loader = torch.utils.data.DataLoader(
        source_test_dataset, batch_size=63, shuffle=False,
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



