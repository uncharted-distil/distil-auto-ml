#!/bin/bash

PROJECT_ROOT=$(pwd)
mkdir -p third_party

pip3 install numpy==1.15.4
pip3 install pandas==0.23.4
pip3 install scipy==1.2.0
pip3 install scikit-learn==0.20.2
pip3 install sklearn_pandas==1.8.0
pip3 install joblib==0.13.0
pip3 install fastdtw==0.3.2
pip3 install networkx==2.2.0
pip3 install pytorch==1.0.0
pip3 install torchvision
pip3 install pytorch
pip3 install tensorflow-gpu==1.12.0
pip3 install fastai

# Recommender system
pip3 install git+https://github.com/bkj/basenet.git@903756540b89809ef458f35257287b937b333417

# Link prediction
pip3 install nose==1.3.7
pip3 install git+https://github.com/mnick/rescal.py

# Audio
cd third_party
git clone https://github.com/tensorflow/models
cd models/research/audioset
pip install resampy==0.2.1
pip install soundfile==0.10.2
curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
cd $PROJECT_ROOT
mv third_party/models/research/audioset third_party/audioset
rm -rf third_party/models/

# Image
# ???

# Pretrained NLP
pip3 install pytorch-pretrained-bert
