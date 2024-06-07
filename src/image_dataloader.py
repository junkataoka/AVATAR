import os
import torchvision.transforms as transforms
from folder import ImageFolder
import numpy as np
import collections

def createSubsetIndice(dataset, majorityClass, ratio):

    indice = [i for i in range(len(dataset))]
    labels = dataset.tgts
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
    return res

def generate_dataset(data_dir, src_data, tar_data, src_domain, tar_domain):
    # Data loading code
    traindir = os.path.join(data_dir, src_data, src_domain)
    traindir_t = os.path.join(data_dir, tar_data, tar_domain)

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
    target_test_dataset = ImageFolder(root=traindir_t, transform=data_transform_test)

    return source_train_dataset, source_test_dataset, target_train_dataset, target_test_dataset
