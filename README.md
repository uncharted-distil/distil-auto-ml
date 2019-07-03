## Distil Auto ML

#### What is this
Mirror of main TA2 system for Uncharted and Qntfy 

Main repo is https://github.com/uncharted-distil/distil-auto-ml

#### Installation

```shell
./build.sh
pip3 install -r server-requirements.txt
```

#### Get those static files 
```shell
$ docker-compose run distil bash 
# cd /static && python3 -m d3m index download
```
