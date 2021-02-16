# Distil Auto ML

Introductory text here. Reference relevant repositories - d3m, primitives, datasets

## Quickstart

Description of shortest path to running the system with some data. This would involve:

> 1.  Getting data sets - we probably need guidance from D3M > program on this since they have been private to date
> 2.  Getting static files - d3m CLI command is `python3 -m d3m > index download`
> 3.  Pulling the TA2 docker image from some place public
> 4.  Running the search command through the container using the > d3m CLI
> 5.  Running the score command on the top ranked pipeline using > the d3m CLI

## Development

### Running From Source

#### Requirements:

> 1. [Python 3.6](https://www.python.org/downloads/release/python-360/)
> 2. Pip (Python 3.6 should come with it)
> 3. [virtualvenv](https://pypi.org/project/virtualenv/)

### Instructions on setting up to run from source:

> - Clone distil-auto-ml
>
> ```console
> git clone https://github.com/uncharted-distil/distil-auto-ml
> ```
> - Install snappy on <b>Linux</b>
> ```console
> sudo apt-get install snappy-dev
> ```
> - Install snappy on <b>MacOS</b>
> ```console
> homebrew install snappy
> ```
> - Clone common-primitives
>
> ```console
>  git clone https://gitlab.com/datadrivendiscovery/common-primitives.git
> ```
>
> - Clone d3m-primitives
>
> ```console
>  git clone https://github.com/cdbethune/d3m-primitives
> ```
>
> - Clone d3m
>
> ```console
>  git clone https://gitlab.com/datadrivendiscovery/d3m
> ```
>
> - Clone distil-primitives
>
> ```console
>  git clone https://github.com/uncharted-distil/distil-primitives
> ```
>
> - Change into the distil-auto-ml directory
>
> ```console
>  cd distil-auto-ml
> ```
>
> - To avoid package collision it is recommended to create a [virtual environment](https://pypi.org/project/virtualenv/)
> - If virtualenv is not installed. Install virtualenv now.
>
> ```console
>  python3 -m pip install virtualenv
> ```
>
> - Create the environment
>
> ```console
>  virtualenv env
> ```
>
> - Activate the environment
>
> ```console
>  source env/bin/activate
> ```
>
> - Installing through server-requirements.txt
>
> ```console
> pip install -r server-requirements.txt
> ```
>
> - Install all the other repository dependencies <b>IMPORTANT:</b> if running on the <b>CPU</b> replace \[gpu\] with \[cpu\]
>
> ```console
>  cd ..
>  cd d3m
>  pip install -e .\[gpu\]
>  cd ..
>  cd d3m-primitives
>  pip install -e .\[gpu\]
>  cd ..
>  cd distil-primitives
>  pip install -e .\[gpu\]
>  cd ..
>  cd common-primitives
>  pip install -e .\[gpu\]
>  pip install python-lzo
>  pip install sent2vec
>  pip install hyppo==0.1.3
>  pip install mxnet
>  pip install -e git+https://github.com/NewKnowledge/simon-d3m-wrapper.git#egg=SimonD3MWrapper
>  pip install -e git+https://gitlab.com/datadrivendiscovery/sklearn-wrap.git@dist#egg=sklearn_wrap
>  pip install -e git+https://github.com/usc-isi-i2/dsbox-primitives#egg=dsbox-primitives
>  pip install -e git+https://github.com/neurodata/primitives-interfaces#egg=jhu-primitives
>   # if error with enum and IntFlag try pip uninstall -y enum34
>   # fix some clobberd versioning
>  pip install -U pandas
> ```
>
> ## MongoDB
> Distil AutoML uses MongoDB as a backend store for it's internal hyperparameter tuning
> There are good instructions depending on your os from the official MongoDB Docs: https://docs.mongodb.com/manual/installation/
>
> - Distil-auto-ml is ready for use 
>```console
> ./run.sh
>```
> - Use [D3M CLI](https://gitlab.com/datadrivendiscovery/d3m) to interface with distil-auto-ml
 
### Building the Docker Container

#### CPU:

> Building a docker image with <b>CPU</b> support is accomplished by invoking the <b>docker_build.sh</b> script:
>
> #### MacOS/Linux
>
> ```console
> $sudo ./docker_build.sh
> ```
>
> #### Windows
>
> Run command prompt as administrator.
>
> ```console
> $./docker_build.sh
> ```

#### GPU:

> Building a docker image with <b>GPU</b> support is accomplished by adding the -g flag to the <b>docker_build.sh</b> call:
>
> #### MacOS/Linux
>
> ```console
> sudo ./docker_build.sh -g
> ```
>
> #### Windows
>
> Run command prompt as administrator.
>
> ```console
> ./docker_build.sh -g
> ```

#### Troubleshooting Docker Image Failing to Build:

> In the event that building the docker image fails and all of the above criteria has been met. One can invoke the <b>docker_build.sh</b> script again this time adding the -f flag. The -f flag forces the download and reinstall of all dependencies regardless of if they meet criteria. <b>Note:</b> if one is building for <b>GPU</b> support - remember the additional -g flag.
>
> #### MacOS/Linux
>
> ```console
> sudo ./docker_build.sh -f
> ```
>
> #### Windows
>
> Run command prompt as administrator.
>
> ```console
> ./docker_build.sh -f
> ```
