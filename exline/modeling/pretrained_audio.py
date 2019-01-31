#!/usr/bin/env python

"""
    exline/modeling/pretrained_audio.py
    
    Featurize audio w/ pretrained audioset model, 
    then train ForestCV model
    
"""

import os
import sys
import numpy as np
from tqdm import tqdm

BASE_PATH = './third_party/audioset'
MODEL_PATH = os.path.join(BASE_PATH, 'vggish_model.ckpt')

sys.path.append(BASE_PATH)
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
import tensorflow as tf

from ..utils import parmap
from .forest import ForestCV
from .metrics import metrics

# --
# Helpers

def audioarray2mel(w):
    assert w.data.dtype == np.int16
    return vggish_input.waveform_to_examples(w.data / 32768.0, w.sample_rate)

def audio2vec(X):
    with tf.Graph().as_default(), tf.Session() as sess:
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, MODEL_PATH)
      
        feat_ = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        emb_  = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
        
        all_feats = []
        for xx in tqdm(X):
            [feats] = sess.run([emb_], feed_dict={feat_: xx})
            all_feats.append(feats)
        
        return all_feats

# --
# Model

class AudiosetModel:
    
    def __init__(self, target_metric):
        self.target_metric = target_metric
    
    def _featurize(self, A):
        mel_feats = parmap(audioarray2mel, A, verbose=10)
        vec_feats = audio2vec(mel_feats)
        return np.vstack([f.max(axis=0) for f in vec_feats])
    
    def fit(self, X_train, y_train):
        vec_maxpool = self._featurize(X_train)
        self.model = ForestCV(target_metric=self.target_metric)
        self.model = self.model.fit(vec_maxpool, y_train)
        return self
    
    def predict(self, X, y):
        vec_maxpool = self._featurize(X)
        return self.model.predict(vec_maxpool)

