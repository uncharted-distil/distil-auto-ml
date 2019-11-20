#!/bin/sh

# arg 1 - directory where pipeline resides
# arg 2 - pipeline ID (don't include .json / .meta)
# arg 3 - dataset location
# arg 4 - dataset name
python3 -m d3m \
    runtime \
    -v ${D3MSTATICDIR} \
    -d ${3} \
    fit-score \
    -r ${3}/${4}/${4}_problem/problemDoc.json \
    -i ${3}/${4}/TRAIN/dataset_TRAIN/datasetDoc.json \
    -t ${3}/${4}/TEST/dataset_TEST/datasetDoc.json \
    -a ${3}/${4}/SCORE/dataset_TEST/datasetDoc.json \
    -p ${1}/${2}.json
