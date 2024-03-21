# Code for "[HAQ: Hardware-Aware Automated Quantization with Mixed Precision"
# Kuan Wang*, Zhijian Liu*, Yujun Lin*, Ji Lin, Song Han
# {kuanwang, zhijian, yujunlin, jilin, songhan}@mit.edu

import os
import numpy as np

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np

def get_cifar10_loaders(batch_size=256, n_worker=32, val_size=10000, train_size=None, random_seed=1, data_root='../Datasets/cifar10/'):
    print('==> Preparing CIFAR-10 data..')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

    trainset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)

    n_train = len(trainset)
    indices = np.arange(n_train)  # Generate indices from 0 to n_train - 1

    # Shuffle the indices with the given random seed
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    assert val_size < n_train, 'val size should be less than n_train'
    train_idx, val_idx = indices[val_size:], indices[:val_size]
    # if train_size:
    #     train_idx = train_idx[:train_size]
    # print('Data: train: {}, val: {}'.format(len(train_idx), len(val_idx)))

    # train_sampler = SubsetRandomSampler(train_idx)
    # val_sampler = SubsetRandomSampler(val_idx)

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )
    n_class = 10

    return train_loader, val_loader, n_class

def get_dataset(dataset_name, batch_size, n_worker, data_root='data/imagenet', for_inception=False):
    print('==> Preparing data..')
    if dataset_name == 'imagenet':
        traindir = os.path.join(data_root, 'train')
        valdir = os.path.join(data_root, 'val')
        assert os.path.exists(traindir), traindir + ' not found'
        assert os.path.exists(valdir), valdir + ' not found'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        input_size = 299 if for_inception else 224

        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                traindir, transforms.Compose([
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=batch_size, shuffle=True,
            num_workers=n_worker, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=n_worker, pin_memory=True)

        n_class = 1000
    elif dataset_name == 'imagenet100':
        traindir = os.path.join(data_root, 'train')
        valdir = os.path.join(data_root, 'val')
        assert os.path.exists(traindir), traindir + ' not found'
        assert os.path.exists(valdir), valdir + ' not found'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        input_size = 299 if for_inception else 224

        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                traindir, transforms.Compose([
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=batch_size, shuffle=True,
            num_workers=n_worker, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=n_worker, pin_memory=True)

        n_class = 100
    elif dataset_name == 'imagenet10':
        traindir = os.path.join(data_root, 'train')
        valdir = os.path.join(data_root, 'val')
        assert os.path.exists(traindir), traindir + ' not found'
        assert os.path.exists(valdir), valdir + ' not found'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        input_size = 299 if for_inception else 224

        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                traindir, transforms.Compose([
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=batch_size, shuffle=True,
            num_workers=n_worker, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=n_worker, pin_memory=True)

        n_class = 10
    else:
        # Add customized data here
        raise NotImplementedError
    return train_loader, val_loader, n_class


def get_split_train_dataset(dataset_name, batch_size, n_worker, val_size, train_size=None, random_seed=1,
                            data_root='data/imagenet', for_inception=False, shuffle=True):
    if shuffle:
        index_sampler = SubsetRandomSampler
    else:
        # use the same order
        class SubsetSequentialSampler(SubsetRandomSampler):
            def __iter__(self):
                return (self.indices[i] for i in torch.arange(len(self.indices)).int())
        index_sampler = SubsetSequentialSampler

    print('==> Preparing data..')
    if dataset_name == 'imagenet' or dataset_name == "ImageNet":

        traindir = os.path.join(data_root, 'train')
        valdir = os.path.join(data_root, 'val')
        assert os.path.exists(traindir), traindir + ' not found'
        assert os.path.exists(valdir), valdir + ' not found'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        input_size = 299 if for_inception else 224
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        test_transform = transforms.Compose([
                transforms.Resize(int(input_size/0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])

        trainset = datasets.ImageFolder(traindir, train_transform)
        valset = datasets.ImageFolder(traindir, test_transform)

        n_train = len(trainset)
        indices = list(range(n_train))
        # shuffle the indices
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        assert val_size < n_train, 'val size should less than n_train'
        train_idx, val_idx = indices[val_size:], indices[:val_size]
        if train_size:
            train_idx = train_idx[:train_size]
        print('Data: train: {}, val: {}'.format(len(train_idx), len(val_idx)))

        train_sampler = index_sampler(train_idx)
        val_sampler = index_sampler(val_idx)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                                                   num_workers=n_worker, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, sampler=val_sampler,
                                                 num_workers=n_worker, pin_memory=True)
        n_class = 1000
    elif dataset_name == 'imagenet100':

        traindir = os.path.join(data_root, 'train')
        valdir = os.path.join(data_root, 'val')
        assert os.path.exists(traindir), traindir + ' not found'
        assert os.path.exists(valdir), valdir + ' not found'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        input_size = 299 if for_inception else 224
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize(int(input_size/0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])

        trainset = datasets.ImageFolder(traindir, train_transform)
        valset = datasets.ImageFolder(traindir, test_transform)

        n_train = len(trainset)
        indices = list(range(n_train))
        # shuffle the indices
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        assert val_size < n_train, 'val size should less than n_train'
        train_idx, val_idx = indices[val_size:], indices[:val_size]
        if train_size:
            train_idx = train_idx[:train_size]
        print('Data: train: {}, val: {}'.format(len(train_idx), len(val_idx)))

        train_sampler = index_sampler(train_idx)
        val_sampler = index_sampler(val_idx)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                                                   num_workers=n_worker, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, sampler=val_sampler,
                                                 num_workers=n_worker, pin_memory=True)
        n_class = 100
    elif dataset_name=="cifar10":
        
        train_loader, val_loader, n_class = get_cifar10_loaders()
    
    
    else:
        raise NotImplementedError

    return train_loader, val_loader, n_class
