#!/bin/bash

pip3 install scipy==1.2.0
pip3 install sklearn_pandas==1.8.0
pip3 install joblib==0.13.0
pip3 install fastdtw==0.3.2
pip3 install networkx==2.2.0
pip3 install torchvision
pip3 install tensorflow-gpu==1.12.0
pip3 install fastai

# Recommender system
pip3 install git+https://github.com/bkj/basenet.git@903756540b89809ef458f35257287b937b333417

# Link prediction
pip3 install nose==1.3.7
pip3 install git+https://github.com/mnick/rescal.py