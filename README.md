# Distil Auto ML

Distil Auto ML is an AutoML system that integrates with [D3M](https://datadrivendiscovery.org/) 

More specifically it is the TA2 system from Uncharted and Qntfy

Main repo is https://github.com/uncharted-distil/distil-auto-ml

## Quickstart using Docker
The TA2 system can be built and started via docker-compose however several static files must be 
downloaded before hand. 

**Datasets** to train on. These may be user created or many examples can be downloaded from https://datasets.datadrivendiscovery.org/d3m/datasets

To train only using the TA2 user generated datasets must be formatted in the same way as the public datasets

**Static Files** may be pretrained weights of a neural network model, or a simple dictionary mapping tokens to necessary ids. 
Pretty much anything *extra* needed to run a ML model within the pipelines. 

To bulk download all static files within the D3M universe **WARNING** this may be quite large
```bash
docker-compose run distil bash 
# cd /static && python3 -m d3m index download
```

One can also pick and choose which static files they wish to download via 

```bash
python3 -m d3m primitive download -p d3m.primitives.path.of.Primitive -o /static
```
For more info on how static files integrate within D3M: https://datadrivendiscovery.org/v2020.11.3/tutorial.html#advanced-primitive-with-static-files

Once the static files and the dataset(s) you want to run on are downloaded

```bash
# symlink your datasets directory 
ln -s ../datasets/seed_datasets_current seed_datasets_current`

# choose the dataset you want to run 
export DATASET=185_baseball

# run it
docker-compose up distil
```

There are two testing TA3 systems also available via docker-compose:
```bash
# run the dummy-ta3 test suite
docker-compose up distil dummy-ta3

# run the simple-ta3 system, which will then be available in the browser at localhost:80
# this requires a directory named 'output' to exist, in addition to the seed_datasets_current directory
docker-compose up distil envoy simple-ta3
```


## Development

### Running From Source

#### Requirements:

> 1. [Python 3.6](https://www.python.org/downloads/release/python-360/)
> 2. Pip (Python 3.6 should come with it)
> 3. [virtualvenv](https://pypi.org/project/virtualenv/)

### Instructions on setting up to run from source:

 - Clone distil-auto-ml

 ```console
 git clone https://github.com/uncharted-distil/distil-auto-ml
 ```
 - Install libraries on <b>Linux</b>
 ```console
 sudo apt-get install snappy-dev build-essential libopenblas-dev libcap-dev ffmpeg
 ```
 - Install libraries on <b>MacOS</b>
 ```console
 brew install snappy cmake openblas libpcap ffmpeg
 ```
 - Clone common-primitives

 ```console
  git clone https://gitlab.com/datadrivendiscovery/common-primitives.git
 ```

 - Clone d3m-primitives

 ```console
  git clone https://github.com/cdbethune/d3m-primitives
 ```

 - Clone d3m

 ```console
  git clone https://gitlab.com/datadrivendiscovery/d3m
 ```

 - Clone distil-primitives

 ```console
  git clone https://github.com/uncharted-distil/distil-primitives
 ```

 - Change into the distil-auto-ml directory

 ```console
  cd distil-auto-ml
 ```

 - To avoid package collision it is recommended to create a [virtual environment](https://pypi.org/project/virtualenv/)
 - If virtualenv is not installed. Install virtualenv now.

 ```console
  python3 -m pip install virtualenv
 ```

 - Create the environment

 ```console
  virtualenv env
 ```

 - Activate the environment

 ```console
  source env/bin/activate
 ```

 - Installing through server-requirements.txt <b>Linux</b>

 ```console
 pip install -r server-requirements.txt
 ```
 - Installing through server-requirements.txt <b>MacOS</b>

 ```console
 CPPFLAGS="-I/usr/local/include -L/usr/local/lib" pip install -r server-requirements.txt
 ```

 - Install all the other repository dependencies <b>IMPORTANT:</b> if running on the <b>CPU</b> replace \[gpu\] with \[cpu\]

 ```console
  cd ..
  cd d3m
  pip install -e .\[gpu\]
  cd ..
  cd d3m-primitives
  pip install -e .\[gpu\]
  cd ..
  cd distil-primitives
  pip install -e .\[gpu\]
  cd ..
  cd common-primitives
  pip install -e .\[gpu\]
  pip install python-lzo
  pip install sent2vec
  pip install hyppo==0.1.3
  pip install mxnet
  pip install -e git+https://github.com/NewKnowledge/simon-d3m-wrapper.git#egg=SimonD3MWrapper
  pip install -e git+https://gitlab.com/datadrivendiscovery/sklearn-wrap.git@dist#egg=sklearn_wrap
  pip install -e git+https://github.com/usc-isi-i2/dsbox-primitives#egg=dsbox-primitives
  pip install -e git+https://github.com/neurodata/primitives-interfaces#egg=jhu-primitives
   # if error with enum and IntFlag try pip uninstall -y enum34
   # fix some clobberd versioning
  pip install -U pandas
 ```

 - MongoDB
 > Distil AutoML uses MongoDB as a backend store for it's internal hyperparameter tuning
 > There are good instructions depending on your os from the official MongoDB Docs: https://docs.mongodb.com/manual/installation/

 - Distil-auto-ml is ready for use 
```console
 ./run.sh
```
 - generate pipelines
```console
 mkdir pipelines
 python3 export_pipelines.sh
```
 - Use [D3M CLI](https://gitlab.com/datadrivendiscovery/d3m) to interface with distil-auto-ml
### Running D3M CLI Example
This section assumes the [source](#Instructions-on-setting-up-to-run-from-source) has been successfully installed and the [datasets](#quickstart-using-docker) have been downloaded.
### Building the Docker Container

#### CPU:

 Building a docker image with <b>CPU</b> support is accomplished by invoking the <b>docker_build.sh</b> script:

 #### MacOS/Linux

 ```console
 sudo ./docker_build.sh
 ```

 #### Windows

 Run command prompt as administrator.

 ```console
 ./docker_build.sh
 ```

#### GPU:

 Building a docker image with <b>GPU</b> support is accomplished by adding the -g flag to the <b>docker_build.sh</b> call:

 #### MacOS/Linux

 ```console
 sudo ./docker_build.sh -g
 ```

 #### Windows

 Run command prompt as administrator.

 ```console
 ./docker_build.sh -g
 ```

#### Troubleshooting Docker Image Failing to Build:

In the event that building the docker image fails and all of the above criteria has been met. One can invoke the <b>docker_build.sh</b> script again this time adding the -f flag. The -f flag forces the download and reinstall of all dependencies regardless of if they meet criteria. <b>Note:</b> if one is building for <b>GPU</b> support - remember the additional -g flag.

 #### MacOS/Linux

 ```console
 sudo ./docker_build.sh -f
 ```

 #### Windows

 Run command prompt as administrator.

 ```console
 ./docker_build.sh -f
 ```
