import torch
import torchvision.datasets as dsets
from torch.utils.data import random_split
import os

from .transform import simple_transform_mnist, simple_transform, cencrop_teransform, imagenet_transform, simple_transform_test, imagenet_transform_aug
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
root = r'./datasets/'
# root = r'/public/home/xiesj/project/data/'
torch.manual_seed(42)
# Set up N clients with 70% training data divided equally
def setup_clients(N, data_set, batch_size, n_worker=0):
    """
    Setup clients for federated learning.
    Args:
    - N: Number of clients.
    - data_set: Name of the dataset to use.
    - batch_size: Batch size for each client.
    - n_worker: Number of workers for data loading.
    
    Returns:
    - clients: A list of data loaders, one for each client.
    """
    
    # Get the full dataset for clients
    #train_loader = get_data(data_set, batch_size=batch_size, n_worker=n_worker, train=True)
    train_loader = get_data(data_set, batch_size=batch_size, n_worker=n_worker, train=True)
    
    # Split the training dataset equally among N clients
    dataset = train_loader.dataset
    #client_datasets = random_split(dataset, [len(dataset) // N for _ in range(N)])
    base_size = len(dataset) // N
    remainder = len(dataset)  % N

# Create the sizes for each client, distributing the remainder
    client_sizes = [base_size + 1 if i < remainder else base_size for i in range(N)]

# Split the dataset into client datasets
    client_datasets = random_split(dataset, client_sizes)
    clients = []
    for i in range(N):
        clients.append(torch.utils.data.DataLoader(client_datasets[i], batch_size=batch_size, shuffle=True, num_workers=n_worker))
    
    return clients
def get_FL_data(data_set, batch_size, shuffle=True, n_worker=0, train = True, add_noise=0):
    if data_set == 'MNIST':
        tran = simple_transform_mnist()
        dataset = dsets.MNIST(root+'MNIST/', train=train, transform=tran, target_transform=None, download=True)
    elif data_set == 'EuroSAT':
        if train:
            tran = simple_transform(64)
        else:
            tran = simple_transform_test(64)
        dataset = dsets.EuroSAT(root+'EuroSAT/', transform=tran, target_transform=None, download=False)   
        train_size = int(0.7 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_size = int(0.3 * len(val_dataset))
        val_size = len(val_dataset) - train_size 
        train_dataset, val_dataset = random_split(val_dataset, [train_size, val_size])        
        if train:
            dataset = train_dataset
        else:
            dataset = val_dataset        
    else:
        print('Sorry! Cannot support ...')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_worker)
    return dataloader
def get_data(data_set, batch_size, shuffle=True, n_worker=0, train = True, add_noise=0):
    if data_set == 'MNIST':
        tran = simple_transform_mnist()
        dataset = dsets.MNIST(root+'MNIST/', train=train, transform=tran, target_transform=None, download=True)
    elif data_set == 'EuroSAT':
        if train:
            tran = simple_transform(64)
        else:
            tran = simple_transform_test(64)
        dataset = dsets.EuroSAT(root+'EuroSAT/', transform=tran, target_transform=None, download=False)   
        train_size = int(0.7 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        if train:
            dataset = train_dataset
        else:
            dataset = val_dataset        
    elif data_set == 'CIFAR10':
        if train:
            tran = simple_transform(32)
        else:
            tran = simple_transform_test(32)
        dataset = dsets.CIFAR10(root+'CIFAR10/', train=train, transform=tran, target_transform=None, download=True)
        
    elif data_set == 'CIFAR100':
        tran = simple_transform(32)
        dataset = dsets.CIFAR100(root+'CIFAR100/', transform=tran, target_transform=None, download=False)
        
    elif data_set == 'CelebA':
        tran = cencrop_teransform(168, resize=(128,128))
        split = 'train' if train else 'test'
        dataset = dsets.CelebA(root+'CelebA/', split=split, transform=tran, target_transform=None, download=False)
    elif data_set == 'STL10':
        tran = simple_transform(96)
        split = 'train+unlabeled' if train else 'test'
        folds = None # For valuation
        dataset = dsets.STL10(root+'STL10/', split=split, folds=folds, transform=tran, target_transform=None, download=False)
    elif data_set == 'Caltech101':
        tran = cencrop_teransform(300, resize=(256,256))
        dataset = dsets.Caltech101(root+'Caltech101', transform=tran, target_transform=None, download=False)
    elif data_set == 'Caltech256':
        tran = cencrop_teransform(168)
        dataset = dsets.Caltech256(root+'Caltech256', transform=tran, target_transform=None, download=False)
    elif data_set == 'Imagenet':
        tran = imagenet_transform(64)
        split = 'train' if train else 'val'
        way = os.path.join(root+'ImageNet/imagenet-mini-100', split)
        dataset = dsets.ImageFolder(way, tran) 
    else:
        print('Sorry! Cannot support ...')
   # print('dataset leng')    
   # print(len(dataset))    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_worker)
    return dataloader


if __name__ == '__main__':
    from utils import play_show
    import matplotlib.pyplot as plt
    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
    cpu = torch.device("cpu")
    dl_1 = get_data('Imagenet', 100, shuffle=True)
    data, _ = next(iter(dl_1))
    print(data.shape)
    print(_)
    play_show(data, device)
    plt.show()
    


        
