import functools
import os
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rde.datasets.skempi_pyrosetta import SkempiPyRosettaDataset
from rde.utils.transforms import get_transform
from rde.utils.data_skempi_mpnn import PaddingCollate
from rde.utils.misc import BlackHole

class SkempiPyRosettaDatasetManager(object):
    """Dataset manager for PyRosetta structures with structure limiting"""

    def __init__(self, config, num_cvfolds, num_workers=4, logger=BlackHole()):
        super().__init__()
        self.config = config
        self.num_cvfolds = num_cvfolds
        self.train_loader = []
        self.val_loaders = []
        self.chains = []
        self.logger = logger
        self.num_workers = num_workers
        
        # Get max_structures from config
        self.max_structures = getattr(config.data, 'max_structures', 4000)
        
        for fold in range(num_cvfolds):
            train_loader, val_loader = self.init_loaders(fold)
            self.train_loader.append(train_loader)
            self.val_loaders.append(val_loader)

    def init_loaders(self, fold):
        config = self.config
        dataset_ = functools.partial(
            SkempiPyRosettaDataset,
            csv_path=config.data.csv_path,
            pdb_wt_dir=config.data.pdb_wt_dir,
            pdb_mt_dir=config.data.pdb_mt_dir,
            cache_dir=config.data.cache_dir,
            num_cvfolds=self.num_cvfolds,
            cvfold_index=fold,
            max_structures=self.max_structures,  # Pass the structure limit
        )

        train_dataset = dataset_(split='train', transform=get_transform(config.data.train.transform))
        val_dataset = dataset_(split='val', transform=get_transform(config.data.val.transform))
        
        train_cplx = set([e['complex'] for e in train_dataset.entries])
        val_cplx = set([e['complex'] for e in val_dataset.entries])
        leakage = train_cplx.intersection(val_cplx)
        assert len(leakage) == 0, f'data leakage {leakage}'

        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.train.batch_size,
            collate_fn=PaddingCollate(config.data.val.transform[1].patch_size),
            shuffle=True,
            num_workers=self.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.train.batch_size,
            shuffle=False,
            collate_fn=PaddingCollate(config.data.val.transform[1].patch_size),
            num_workers=self.num_workers
        )

        self.logger.info('Fold %d: Train %d, Val %d' % (fold, len(train_dataset), len(val_dataset)))
        return train_loader, val_loader

    def get_train_loader(self, fold):
        return self.train_loader[fold]

    def get_val_loader(self, fold):
        return self.val_loaders[fold] 