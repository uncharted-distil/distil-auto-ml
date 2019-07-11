#!/bin/sh

# arg 1 - directory where pipeline resides
# arg 2 - pipeline ID (don't include .json / .meta)
# arg 3 - dataset directory

docker run \
    --rm \
    --env D3MINPUTDIR=${1} \
    -v ${1}:${1} \
    -v ${3}:${3} \
    -v ${D3MSTATICDIR}:${D3MSTATICDIR} \
    registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-bionic-python36-v2019.6.7 \
    python3 -m d3m runtime -v ${D3MSTATICDIR} -d ${3} fit-score -m ${1}/${2}.meta -p ${1}/${2}.json
