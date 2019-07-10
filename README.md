## Distil Auto ML

#### What is this
Mirror of main TA2 system for Uncharted and Qntfy 

Main repo is https://github.com/uncharted-distil/distil-auto-ml

#### Get this running locally with docker 

* symlink your datasets directory 

`ln -s ../datasets/seed_datasets_current seed_datasets_current`

* choose the dataset you want to run 

`export DATASET=185_baseball`

* run it all 

`docker-compose up`

#### Get those static files 
```shell
$ docker-compose run distil bash 
# cd /static && python3 -m d3m index download
```
