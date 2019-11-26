#!/bin/sh

# arg 1 - path to pipeline json
# arg 2 - dataset name

python3 -m d3m \
    runtime \
    -v ${D3MSTATICDIR} \
    -d ${D3MINPUTDIR} \
    fit-score \
    -r ${D3MINPUTDIR}/${2}/${2}_problem/problemDoc.json \
    -i ${D3MINPUTDIR}/${2}/TRAIN/dataset_TRAIN/datasetDoc.json \
    -t ${D3MINPUTDIR}/${2}/TEST/dataset_TEST/datasetDoc.json \
    -a ${D3MINPUTDIR}/${2}/SCORE/dataset_SCORE/datasetDoc.json \
    -p ${1}
