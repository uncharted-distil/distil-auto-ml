#!/usr/bin/env python

"""
    exline/modeling/fixed_cnn.py
"""

import sys
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms
from torchvision.datasets.folder import pil_loader
from torchvision.models import resnet18, resnet34, resnet50, resnet101, densenet161

from basenet import BaseNet

from .forest import ForestCV
from .helpers import tiebreaking_vote
from .metrics import metrics, classification_metrics, regression_metrics

# --
# IO helper

class PathDataset(Dataset):
    def __init__(self, paths, transform=None, preload=True):
        self.paths   = paths
        
        self.preload = preload
        if self.preload:
            print('PathDataset: preloading', file=sys.stderr)
            self._samples = []
            for p in tqdm(self.paths):
                self._samples.append(pil_loader(p))
        
        self.transform = transform
    
    def __getitem__(self, idx):
        
        if not self.preload:
            sample = pil_loader(self.paths[idx])
        else:
            sample = self._samples[idx]
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample, -1
    
    def __len__(self):
        return self.paths.shape[0]

# --
# Model

class FixedCNNFeatureExtractor(BaseNet):
    def __init__(self, base_model, drop_last=1):
        super().__init__()
        self._model = nn.Sequential(*list(base_model.children())[:-drop_last])
    
    def forward(self, x):
        x = self._model(x)
        while len(x.shape) > 2:
            x = x.mean(dim=-1).squeeze()
        
        return x


class FixedCNNForest:
    
    def __init__(self, target_metric, feature_extractors=[resnet18, resnet34, resnet50, resnet101, densenet161]):
        
        self.target_metric      = target_metric
        self.is_classification  = target_metric in classification_metrics
        self.feature_extractors = feature_extractors
        
    def _fe_fit_predict(self, fe, dataloaders, y_train, y_test):
        model = FixedCNNFeatureExtractor(fe(pretrained=True)).to('cuda')
        model.verbose = True
        _ = model.eval()
        
        train_feats, _ = model.predict(dataloaders, mode='train')
        test_feats, _  = model.predict(dataloaders, mode='test')
        
        forest = ForestCV(
            target_metric=self.target_metric, 
            estimator=['RandomForest'] if self.is_classification else ['ExtraTrees', 'RandomForest']
        )
        forest = forest.fit(train_feats, y_train)
        return forest.predict(test_feats)
    
    def fit_score(self, paths_train, paths_test, y_train, y_test):
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        train_dataset = PathDataset(paths=paths_train, transform=transform)
        test_dataset  = PathDataset(paths=paths_test, transform=transform)
        
        dataloaders = {
            "train" : DataLoader(
                train_dataset,
                batch_size=32,
                shuffle=False,
            ),
            "test"  : DataLoader(
                test_dataset,
                batch_size=32,
                shuffle=False,
            ),
        }
        
        all_preds = []
        for fe in self.feature_extractors:
            test_pred = self._fe_fit_predict(fe, dataloaders, y_train, y_test)
            # >>
            print('score=%f' % metrics[self.target_metric](y_test, test_pred), file=sys.stderr)
            # <<
            all_preds.append(test_pred)
        
        if self.is_classification:
            ens_pred = tiebreaking_vote(np.vstack(all_preds), y_train)
        else:
            ens_pred = np.stack(all_preds).mean(axis=0)
        
        return metrics[self.target_metric](y_test, ens_pred)



