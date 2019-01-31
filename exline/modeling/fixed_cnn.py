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
    
    def __init__(self, target_metric):
        
        self.target_metric      = target_metric
        self.is_classification  = target_metric in classification_metrics
        
        self._feature_extractors = [resnet18, resnet34, resnet50, resnet101, densenet161]
        self._y_train            = None
        self._models             = None
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def extract_features(self, fe, dataloaders, mode):
        model = FixedCNNFeatureExtractor(fe(pretrained=True)).to('cuda')
        model.verbose = True
        _ = model.eval()
        
        feats, _ = model.predict(dataloaders, mode=mode)
        del model
        
        return feats
    
    def fit(self, X_train, y_train):
        self._y_train = y_train
        
        dataloaders = {
            "train" : DataLoader(
                PathDataset(paths=X_train, transform=self.transform),
                batch_size=32,
                shuffle=False,
            ),
        }
        
        self._models = []
        for fe in self._feature_extractors:
            train_feats = self.extract_features(fe, dataloaders, mode='train')
            model = ForestCV(
                target_metric=self.target_metric, 
                estimator=['RandomForest'] if self.is_classification else ['ExtraTrees', 'RandomForest']
            )
            model = model.fit(train_feats, y_train)
            self._models.append(model)
        
        return self
        
    def predict(self, X):
        dataloaders = {
            "test"  : DataLoader(
                PathDataset(paths=X, transform=self.transform),
                batch_size=32,
                shuffle=False,
            ),
        }
        
        all_preds = []
        for fe, model in zip(self._feature_extractors, self._models):
            test_feats = self.extract_features(fe, dataloaders, mode='test')
            all_preds.append(model.predict(test_feats))
        
        if self.is_classification:
            ens_pred = tiebreaking_vote(np.vstack(all_preds), self._y_train)
        else:
            ens_pred = np.stack(all_preds).mean(axis=0)
        
        return ens_pred