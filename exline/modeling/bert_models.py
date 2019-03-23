#!/usr/bin/env python

"""
    exline/modeling/bert_models.py
"""

import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

"""
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel, PreTrainedBertModel, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
"""

from .base import EXLineBaseModel
from .metrics import metrics, classification_metrics

# --
# Helpers

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
    
    return tokens_a, tokens_b


def examples2dataset(examples, label_list, max_seq_length, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example['text_a'])
        tokens_b = tokenizer.tokenize(example['text_b'])
        tokens_a, tokens_b = _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        
        tokens      = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        segment_ids = ([0] * (len(tokens_a) + 2)) + ([1] * (len(tokens_b) + 1))
        
        input_ids  = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids   += padding
        input_mask  += padding
        segment_ids += padding
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        features.append({
            "input_ids"   : input_ids,
            "input_mask"  : input_mask,
            "segment_ids" : segment_ids,
            "label_id"    : example['label'],
        })
    
    all_input_ids   = torch.LongTensor([f['input_ids'] for f in features])
    all_input_mask  = torch.LongTensor([f['input_mask'] for f in features])
    all_segment_ids = torch.LongTensor([f['segment_ids'] for f in features])
    all_label_ids   = torch.LongTensor([f['label_id'] for f in features])
    
    return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

# --
# Model helpers

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    else:
        return 1.0 - x

"""
class QAModel(PreTrainedBertModel):
    def __init__(self, config, num_labels=2, weights=None):
        super().__init__(config)
        
        self.num_labels = num_labels
        self.bert       = BertModel(config)
        self.dropout    = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)
        
        if weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(torch.FloatTensor(weights))
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        
        self.use_classifier = True
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        
        if not self.use_classifier:
            return pooled_output
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            return logits, loss
        else:
            return logits
"""
# --
# Wrapper

class BERTPairClassification(EXLineBaseModel):
    
    def __init__(self, target_metric, columns=['question', 'sentence'], 
        batch_size=32, learning_rate=5e-5, epochs=3, warmup_proportion=0.1, seed=123):
        
        assert target_metric in classification_metrics
        
        self.target_metric = target_metric
        self.columns       = columns
        
        self.batch_size        = batch_size
        self.learning_rate     = learning_rate
        self.epochs            = epochs
        self.warmup_proportion = warmup_proportion
        
        self.bert_model        = 'bert-base-uncased'
        self.do_lower_case     = True
        self.device            = torch.device("cuda")
        
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=self.do_lower_case)
        
        _ = np.random.seed(seed)
        _ = torch.manual_seed(seed + 1)
        _ = torch.cuda.manual_seed_all(seed + 2)
        
    def _set_lr(self, progress):
        lr_this_step = self.learning_rate * warmup_linear(progress, self.warmup_proportion)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_this_step
            
    def _row2example(self, row):
        return {
            "text_a" : row[self.columns[0]],
            "text_b" : row[self.columns[1]],
            "label"  : int(row['_label']),
        }
    
    def fit(self, X_train, y_train, U_train=None):
        
        assert 'question' in X_train.columns
        assert 'sentence' in X_train.columns
        
        # --
        # Prep
        
        X_train = X_train.copy()
        X_train['_label'] = y_train
        
        train_examples = list(X_train.apply(self._row2example, axis=1))
        
        self.label_list = list(set(X_train._label.astype(str)))
        self.num_labels = len(self.label_list)
        num_train_steps = int(len(train_examples) / self.batch_size * float(self.epochs))
        
        q_lens = X_train.question.apply(lambda x: len(self.tokenizer.tokenize(x)))
        s_lens = X_train.sentence.apply(lambda x: len(self.tokenizer.tokenize(x)))
        self.max_seq_len = int(np.percentile(q_lens + s_lens, 99) + 1)
        
        train_dataset = examples2dataset(train_examples, self.label_list, self.max_seq_len, self.tokenizer)
        
        dataloaders = {
            "train" : DataLoader(
                dataset=train_dataset, 
                shuffle=True,
                batch_size=self.batch_size,
                num_workers=4,
            ),
        }
        
        # --
        # Define model
        
        self.model = QAModel.from_pretrained(
            self.bert_model,
            num_labels=self.num_labels,
            # weights=[0.1, 1],
        ).to(self.device)
        
        # --
        # Optimizer
        
        params         = list(self.model.named_parameters())
        no_decay       = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_params = [
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        
        self.optimizer = BertAdam(
            params=grouped_params,
            lr=self.learning_rate,
            warmup=self.warmup_proportion,
            t_total=num_train_steps,
        )
        
        # --
        # Train
        
        train_step = 0
        _ = self.model.train()
        for epoch_idx in tqdm(range(self.epochs), desc="Epoch"):
            
            train_loss_hist = []
            gen = tqdm(dataloaders['train'], desc="train iter")
            for step, batch in enumerate(gen):
                input_ids, input_mask, segment_ids, label_ids = tuple(t.to(self.device) for t in batch)
                
                _, loss = self.model(input_ids, segment_ids, input_mask, label_ids)
                loss.backward()
                
                train_loss_hist.append(loss.item())
                
                self._set_lr(train_step / num_train_steps)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_step += 1
                
                gen.set_postfix(loss=loss.item())
        
        self.train_loss_hist = train_loss_hist
        
        return self
        
    def predict(self, X):
        
        # --
        # Prep
        
        X = X.copy()
        X['_label'] = -1
        
        examples = list(X.apply(self._row2example, axis=1))
        
        dataset = examples2dataset(examples, self.label_list, self.max_seq_len, self.tokenizer)
        dataloaders = {
            "test" : list(DataLoader(
                dataset=dataset, 
                shuffle=False,
                batch_size=self.batch_size,
                num_workers=4,
            )),
        }
        
        # --
        # Predict
        
        _ = self.model.eval()
        all_logits = []
        gen = tqdm(dataloaders['test'], desc="score iter")
        for step, batch in enumerate(gen):
            
            input_ids, input_mask, segment_ids, _ = tuple(t.to(self.device) for t in batch)
            
            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
            
            logits = logits.detach().cpu().numpy()
            all_logits.append(logits)
        
        return np.vstack(all_logits).argmax(axis=-1)

