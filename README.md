## exline

#### Installation

```
conda create -n exline_env python=3.6 pip -y
source activate exline_env

PROJECT_ROOT=$(pwd)
mkdir -p third_party

pip install numpy==1.15.4
pip install pandas==0.23.4
pip install scipy==1.2.0
pip install scikit-learn==0.20.2
pip install sklearn_pandas==1.8.0
pip install joblib==0.13.0
pip install fastdtw==0.3.2
pip install networkx==2.2.0
conda install pytorch==1.0.0 torchvision -c pytorch -y
conda install -c anaconda tensorflow-gpu==1.12.0 -y

# SGM
cd third_party
git clone https://github.com/bkj/sgm.git
cd sgm
pip install cython==0.29.3
pip install -r requirements.txt
pip install -e .
cd $PROJECT_ROOT

# Recommender system
pip install git+https://github.com/bkj/basenet.git@903756540b89809ef458f35257287b937b333417

# Link prediction
pip install nose==1.3.7
pip install git+https://github.com/mnick/rescal.py

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
# conda install -c pytorch -c fastai fastai -y

# Pretrained NLP
pip install pytorch-pretrained-bert

# install exline
pip install -e .
```

#### D3M Data Download

```
git lfs clone git@gitlab.datadrivendiscovery.org:d3m/datasets.git
ln -s datasets exline/d3m_datasets
# ln -s /raid/users/bjohnson/projects/sdh/d3m_datasets d3m_datasets
```

#### Usage

```
python -m exline.main --prob-name $PROB_NAME
```
