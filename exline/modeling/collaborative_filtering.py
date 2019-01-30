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

from .metrics import metrics, classification_metrics, regression_metrics

# --
# Helpers

def prep_cf(X_train, X_test):
    
    assert X_train.shape[1] == 2
    assert X_test.shape[1] == 2
    
    X_train = X_train.copy()
    X_test  = X_test.copy()
    
    X_train.columns = ('user', 'item')
    X_test.columns  = ('user', 'item')
    
    uusers       = np.unique(np.hstack([X_train.user, X_test.user]))
    user_lookup  = dict(zip(uusers, range(len(uusers))))
    X_train.user = X_train.user.apply(user_lookup.get)
    X_test.user  = X_test.user.apply(user_lookup.get)
    
    uitems       = np.unique(np.hstack([X_train.item, X_test.item]))
    item_lookup  = dict(zip(uitems, range(len(uitems))))
    X_train.item = X_train.item.apply(item_lookup.get)
    X_test.item  = X_test.item.apply(item_lookup.get)
    
    n_users = len(uusers)
    n_items = len(uitems)
    
    return X_train, X_test, n_users, n_items


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


def make_cf_model(
        loss_fn,
        n_users,
        n_items,
        emb_dim,
        n_outputs,
        lr_max=2e-3,
        epochs=8,
        verbose=False,
    ):
    
    model = CFModel(
        loss_fn=loss_fn,
        n_users=n_users,
        n_items=n_items,
        emb_dim=emb_dim,
        n_outputs=n_outputs,
    )
    
    model.verbose = verbose
    
    lr_scheduler = HPSchedule.linear(hp_max=lr_max, epochs=epochs)
    model.init_optimizer(
        opt=torch.optim.Adam,
        params=model.parameters(),
        hp_scheduler={"lr" : lr_scheduler},
    )
    
    return model


class SGDCollaborativeFilter:
    
    def __init__(self, target_metric, emb_dims=[128, 256, 512, 1024], n_outputs=1,
        epochs=8, batch_size=512, device='cuda'):
        
        if target_metric == 'meanAbsoluteError':
            self.loss_fn = F.l1_loss
        elif target_metric == 'accuracy':
            self.loss_fn = F.binary_cross_entropy_with_logits
        else:
            raise Exception
        
        self.target_metric = target_metric
        self.emb_dims      = emb_dims
        self.n_outputs     = n_outputs
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.device        = device
    
    def fit_score(self, X_train, X_test, y_train, y_test):
        
        X_train, X_test, n_users, n_items = prep_cf(X_train, X_test)
        
        self.models = [
            make_cf_model(
                loss_fn=self.loss_fn,
                n_users=n_users,
                n_items=n_items,
                emb_dim=emb_dim,
                n_outputs=self.n_outputs,
            ) for emb_dim in self.emb_dims
        ]
        
        dataloaders = {
            "train" : DataLoader(
                TensorDataset(
                    torch.LongTensor(X_train.values),
                    torch.FloatTensor(y_train).view(-1, 1),
                ),
                shuffle=True,
                batch_size=self.batch_size,
            ),
            "test" : DataLoader(
                TensorDataset(
                    torch.LongTensor(X_test.values),
                    torch.FloatTensor(y_test).view(-1, 1),
                ),
                shuffle=False,
                batch_size=self.batch_size,
            )
        }
        
        # --
        # Train
        
        for i, model in enumerate(self.models):
            print('model=%d' % i, file=sys.stderr)
            model = model.to(self.device)
            
            for epoch in range(self.epochs):
                train = model.train_epoch(dataloaders, mode='train', compute_acc=False)
                print({
                    "epoch"      : int(epoch),
                    "train_loss" : float(np.mean(train['loss'])),
                }, file=sys.stderr)
            
            model = model.to('cpu')
        
        # --
        # Test
        
        all_preds = []
        for model in self.models:
            model = model.to(self.device)
            
            preds, _ = model.predict(dataloaders, mode='test')
            all_preds.append(to_numpy(preds).squeeze())
            
            model = model.to('cpu')
        
        self.all_scores = [metrics[self.target_metric](y_test, p) for p in all_preds]
        
        ens_pred  = np.vstack(all_preds).mean(axis=0)
        if self.target_metric in regression_metrics:
            ens_score = metrics[self.target_metric](y_test, ens_pred)
        elif self.target_metric in classification_metrics:
            raise NotImplemented
        else:
            raise Exception
        
        return ens_score
