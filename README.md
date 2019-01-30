## exline

#### Installation

```
conda create -n exline_env python=3.6 pip -y
source activate exline_env

mkdir -p third_party

pip install numpy==1.15.4
pip install pandas==0.23.4
pip install scipy==1.1.0
pip install scikit-learn==0.20.2
pip install sklearn_pandas==1.8.0
pip install joblib==0.13.0
pip install fastdtw==0.3.2
pip install networkx==2.2.0
conda install pytorch==1.0.0 torchvision -c pytorch
conda install -c anaconda tensorflow-gpu==1.12.0

# SGM
PROJECT_ROOT=$(pwd)
cd $PROJECT_ROOT
git clone https://github.com/bkj/sgm.git
cd sgm
pip install cython
pip install -r requirements.txt
pip install -e .
cd $PROJECT_ROOT

# Recommender system
pip install git+https://github.com/bkj/basenet.git@903756540b89809ef458f35257287b937b333417

# Link prediction
pip install nose
pip install git+https://github.com/mnick/rescal.py

# Audio
PROJECT_ROOT=$(pwd)
cd third_party
git clone https://github.com/tensorflow/models
cd models/research/audioset
pip install resampy==0.2.1
pip install soundfile==0.10.2
curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
cd $PROJECT_ROOT

# Image
conda install -c pytorch -c fastai fastai

# Pretrained NLP
pip install pytorch-pretrained-bert

# install exline
pip install -e .

pip install git+https://gitlab.com/datadrivendiscovery/d3m
```

#### D3M Data Download

```
git lfs clone git@gitlab.datadrivendiscovery.org:d3m/datasets.git
ln -s datasets exline/d3m_datasets
```

#### Usage

```
python -m exline.main --prob-name $PROB_NAME
```
