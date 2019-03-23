#!/usr/bin/env python

"""
    exline/modeling/collaborative_filtering.py
"""

import sys

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from basenet import BaseNet, HPSchedule
from basenet.helpers import to_numpy

from .base import EXLineBaseModel
from .metrics import metrics, classification_metrics, regression_metrics

# --
# Models

class CFModel(BaseNet):
    def __init__(self, loss_fn, n_users, n_items, emb_dim=1024, n_outputs=1):
        super().__init__(loss_fn=loss_fn)
        
        self.emb_users = nn.Embedding(n_users, emb_dim)
        self.emb_items = nn.Embedding(n_items, emb_dim)
        self.emb_users.weight.data.uniform_(-0.05, 0.05)
        self.emb_items.weight.data.uniform_(-0.05, 0.05)
        
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)
        
        self.hidden = nn.Linear(2 * emb_dim, emb_dim)
        self.score  = nn.Linear(emb_dim, n_outputs, bias=False)
    
    def forward(self, x):
        users, items = x[:,0], x[:,1]
        user_emb = self.emb_users(users)
        item_emb = self.emb_items(items)
        
        # ?? Dropout
        
        emb = torch.cat([user_emb, item_emb], dim=1)
        emb = self.hidden(emb)
        emb = F.relu(emb)
        return self.score(emb) + self.user_bias(users) + self.item_bias(items)


class SGDCollaborativeFilter(EXLineBaseModel):
    
    def __init__(self, n_users, n_items, target_metric, emb_dims=[128, 256, 512, 1024], n_outputs=1,
        epochs=8, batch_size=512, lr_max=2e-3, device='cuda'):
        
        if target_metric == 'meanAbsoluteError':
            self.loss_fn = F.l1_loss
        # elif target_metric == 'accuracy':
            # self.loss_fn = F.binary_cross_entropy_with_logits
        else:
            raise Exception('SGDCollaborativeFilter: unknown metric')
        
        self.n_users       = n_users
        self.n_items       = n_items
        self.target_metric = target_metric
        self.emb_dims      = emb_dims
        self.n_outputs     = n_outputs
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.device        = device
        self.lr_max        = lr_max
        
    def _make_model(self, emb_dim):
        
        model = CFModel(
            emb_dim=emb_dim,
            loss_fn=self.loss_fn,
            n_users=self.n_users,
            n_items=self.n_items,
            n_outputs=self.n_outputs,
        )
        
        model.init_optimizer(
            opt=torch.optim.Adam,
            params=model.parameters(),
            hp_scheduler={"lr" : HPSchedule.linear(hp_max=self.lr_max, epochs=self.epochs)},
        )
        
        return model
        
    def fit(self, X_train, y_train, U_train=None):
        
        dataloaders = {
            "train" : DataLoader(
                TensorDataset(
                    torch.LongTensor(X_train.values),
                    torch.FloatTensor(y_train).view(-1, 1),
                ),
                shuffle=True,
                batch_size=self.batch_size,
            ),
        }
        
        # --
        # Train
        
        self._models = [self._make_model(emb_dim=emb_dim) for emb_dim in self.emb_dims]
        
        for i, model in enumerate(self._models):
            print('model=%d' % i, file=sys.stderr)
            model = model.to(self.device)
            
            for epoch in range(self.epochs):
                train = model.train_epoch(dataloaders, mode='train', compute_acc=False)
                print({
                    "epoch"      : int(epoch),
                    "train_loss" : float(np.mean(train['loss'])),
                }, file=sys.stderr)
            
            model = model.to('cpu')
        
        return self
    
    def predict(self, X):
        
        dataloaders = {
            "test" : DataLoader(
                TensorDataset(
                    torch.LongTensor(X.values),
                    torch.FloatTensor(np.zeros(X.shape[0]) - 1).view(-1, 1),
                ),
                shuffle=False,
                batch_size=self.batch_size,
            )
        }
        
        # --
        # Test
        
        all_preds = []
        for model in self._models:
            model = model.to(self.device)
            
            preds, _ = model.predict(dataloaders, mode='test')
            all_preds.append(to_numpy(preds).squeeze())
            
            model = model.to('cpu')
        
        return np.vstack(all_preds).mean(axis=0)
