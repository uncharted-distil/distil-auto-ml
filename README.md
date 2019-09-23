# Distil Auto ML

## Summary
TA2 system for Uncharted and Qntfy 

Main repo is https://github.com/uncharted-distil/distil-auto-ml

## How to run
The TA2 system can be built and started via docker-compose:
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

## Get those static files 
```bash
docker-compose run distil bash 
# cd /static && python3 -m d3m index download
```
