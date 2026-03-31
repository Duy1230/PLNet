import torch
from .transforms import *
from . import train_dataset
from ..config.paths_catalog import DatasetCatalog
from . import test_dataset
from ..model.hafm import HAFMencoder


def _uses_raw_superpoint_input(cfg):
    return cfg.MODEL.NAME in {"PointLine", "EnhancedPointLine"}


def build_transform(cfg):
    transforms = Compose(
        [ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                     cfg.DATASETS.IMAGE.WIDTH),
         ToTensor(),
         Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                                           cfg.DATASETS.IMAGE.PIXEL_STD,
                                           cfg.DATASETS.IMAGE.TO_255)
        ]
    )

    if _uses_raw_superpoint_input(cfg):
        transforms = Compose(
            [ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                        cfg.DATASETS.IMAGE.WIDTH),
            ToTensor()
            ]
        )

    return transforms
def build_train_dataset(cfg):
    assert len(cfg.DATASETS.TRAIN) == 1
    name = cfg.DATASETS.TRAIN[0]
    dargs = DatasetCatalog.get(name)

    factory = getattr(train_dataset,dargs['factory'])
    args = dargs['args']
    args['augmentation'] = cfg.DATASETS.AUGMENTATION
    if bool(getattr(cfg.DATALOADER, "PRECOMPUTE_HAFM", True)):
        args['hafm_encoder'] = HAFMencoder(cfg)
    else:
        args['hafm_encoder'] = None
    args['transform'] = Compose(
                                [Resize(cfg.DATASETS.IMAGE.HEIGHT,
                                        cfg.DATASETS.IMAGE.WIDTH,
                                        cfg.DATASETS.TARGET.HEIGHT,
                                        cfg.DATASETS.TARGET.WIDTH),
                                 ToTensor(),
                                 Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                                           cfg.DATASETS.IMAGE.PIXEL_STD,
                                           cfg.DATASETS.IMAGE.TO_255)])

    if _uses_raw_superpoint_input(cfg):
        args['transform'] = Compose(
                                    [Resize(cfg.DATASETS.IMAGE.HEIGHT,
                                            cfg.DATASETS.IMAGE.WIDTH,
                                            cfg.DATASETS.TARGET.HEIGHT,
                                            cfg.DATASETS.TARGET.WIDTH),
                                    ToTensor()])


    dataset = factory(**args)

    num_workers = int(cfg.DATALOADER.NUM_WORKERS)
    pin_memory = bool(getattr(cfg.DATALOADER, "PIN_MEMORY", True))
    persistent_workers = bool(getattr(cfg.DATALOADER, "PERSISTENT_WORKERS", True))
    prefetch_factor = int(getattr(cfg.DATALOADER, "PREFETCH_FACTOR", 2))

    loader_kwargs = {
        "batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "collate_fn": train_dataset.collate_fn,
        "shuffle": True,
        "num_workers": num_workers,
        # Pinned memory enables faster async host-to-device copies.
        "pin_memory": pin_memory,
    }
    if num_workers > 0 and persistent_workers:
        # Keep workers alive across epochs and prefetch batches.
        loader_kwargs["persistent_workers"] = True
    if num_workers > 0 and prefetch_factor > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    dataset = torch.utils.data.DataLoader(dataset, **loader_kwargs)
    return dataset

def build_test_dataset(cfg):
    transforms = Compose(
        [ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                     cfg.DATASETS.IMAGE.WIDTH),
         ToTensor(),
         Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                                           cfg.DATASETS.IMAGE.PIXEL_STD,
                                           cfg.DATASETS.IMAGE.TO_255)
        ]
    )

    if _uses_raw_superpoint_input(cfg):
        transforms = Compose(
            [ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                        cfg.DATASETS.IMAGE.WIDTH),
            ToTensor()
            ]
        )

    datasets = []
    for name in cfg.DATASETS.TEST:
        dargs = DatasetCatalog.get(name)
        factory = getattr(test_dataset,dargs['factory'])
        args = dargs['args']
        args['transform'] = transforms
        dataset = factory(**args)
        num_workers = int(cfg.DATALOADER.NUM_WORKERS)
        pin_memory = bool(getattr(cfg.DATALOADER, "PIN_MEMORY", True))
        persistent_workers = bool(getattr(cfg.DATALOADER, "PERSISTENT_WORKERS", True))
        prefetch_factor = int(getattr(cfg.DATALOADER, "PREFETCH_FACTOR", 2))
        loader_kwargs = {
            "batch_size": 1,
            "collate_fn": dataset.collate_fn,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        if num_workers > 0 and persistent_workers:
            loader_kwargs["persistent_workers"] = True
        if num_workers > 0 and prefetch_factor > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor

        dataset = torch.utils.data.DataLoader(dataset, **loader_kwargs)
        datasets.append((name,dataset))
    return datasets
