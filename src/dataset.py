'''
https://github.com/sthalles/SimCLR/blob/master/data_aug/contrastive_learning_dataset.py
'''

import torchvision
from torchvision import transforms


class ViewGenerator():
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.n_views)]


def get_simclr_transform(size, s=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            transforms.GaussianBlur(kernel_size=int(0.1 * size)),
                                            transforms.ToTensor()])
    return data_transforms


def get_dataset(root_folder, name, n_views):
    valid_datasets = {'cifar10': lambda: torchvision.datasets.CIFAR10(
                                            root_folder,
                                            train=True,
                                            transform=ViewGenerator(get_simclr_transform(32), n_views),
                                            download=True
                                        ),

                        'stl10': lambda: torchvision.datasets.STL10(root_folder,
                                            split='unlabeled',
                                            transform=ViewGenerator(get_simclr_transform(96), n_views),
                                            download=True)
                }


    dataset_fn = valid_datasets[name.lower()]
    return dataset_fn()




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = get_dataset(root_folder='./data', name='cifar10', n_views=2)
    print(dataset)