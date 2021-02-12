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

> - Clone the repository
>
> ```console
>   $ git clone https://github.com/uncharted-distil/distil-auto-ml
> ```
>
> - Change into the distil-auto-ml directory
>
> ```console
>   $ cd distil-auto-ml
> ```
>
> - To avoid package collision it is recommended to create a [virtual environment](https://pypi.org/project/virtualenv/)
> - If virtualenv is not installed. Install virtualenv now.
>
> ```console
>   $ python3 -m pip install virtualenv
> ```
>
> - Create the environment
>
> ```console
>   $ virtualenv env
> ```
>
> - Activate the environment
>
> ```console
>   $ source env/bin/activate
> ```
>
> - Installing through requirements.txt
>
> ```console
>  $ pip install -r requirements.txt
> ```
>
> - Distil-auto-ml is ready for use
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
> $sudo ./docker_build.sh -g
> ```
>
> #### Windows
>
> Run command prompt as administrator.
>
> ```console
> $./docker_build.sh -g
> ```

#### Troubleshooting Docker Image Failing to Build:

> In the event that building the docker image fails and all of the above criteria has been met. One can invoke the <b>docker_build.sh</b> script again this time adding the -f flag. The -f flag forces the download and reinstall of all dependencies regardless of if they meet criteria. <b>Note:</b> if one is building for <b>GPU</b> support - remember the additional -g flag.
>
> #### MacOS/Linux
>
> ```console
> $sudo ./docker_build.sh -f
> ```
>
> #### Windows
>
> Run command prompt as administrator.
>
> ```console
> $./docker_build.sh -f
> ```
