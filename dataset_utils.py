import os
import torch
import multiprocessing

import config


def load_datasets(DatasetClass, transforms=None):
    datasets = {x: DatasetClass(os.path.join(config.DATA_DIR, x), x, transforms) for x in ['train', 'val']}
    dataset_loaders = {
        x: torch.utils.data.DataLoader(datasets[x], batch_size=config.BATCH_SIZE, shuffle=True,
                                       num_workers=multiprocessing.cpu_count()) for x in ['train', 'val']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    return dataset_loaders, dataset_sizes


def load_testset(DatasetClass, transforms=None):
    datasets = {x: DatasetClass(os.path.join(config.DATA_DIR, x), x, transforms) for x in ['test']}
    dataset_loaders = {
        x: torch.utils.data.DataLoader(datasets[x], batch_size=config.BATCH_SIZE * 4, shuffle=False,
                                       num_workers=multiprocessing.cpu_count()) for x in ['test']}
    dataset_sizes = {x: len(datasets[x]) for x in ['test']}
    return dataset_loaders, dataset_sizes
