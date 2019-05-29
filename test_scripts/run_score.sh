#!/bin/bash

PIPELINE_ID=$1
DATASET=$2

[ -z "$D3MINPUTDIR" ] && echo "Error - D3MINPUTDIR unset" && exit 1
[ -z "$D3MOUTPUTDIR" ] && echo "Error - D3MOUTPUTDIR unset" && exit 1
[ -z "$D3MINPUTDIR" ] && echo "Error - D3MSTATICDIR unset" && exit 1
[ -z "$DATASET" ] && echo "Error - DATASET unset" && exit 1
[ -z "$PIPELINE_ID" ] && echo "Error - PIPELINE_ID unset" && exit 1

mkdir -p ${D3MOUTPUTDIR}/logs && \
mkdir -p ${D3MOUTPUTDIR}/predictions && \
mkdir -p ${D3MOUTPUTDIR}/pipeline_runs && \
mkdir -p ${D3MOUTPUTDIR}/score

cp ./scoring_pipeline.yml ${D3MOUTPUTDIR}

python3 -m d3m.runtime \
    --volumes "${D3MSTATICDIR}" \
    --context TESTING \
    fit-score \
    -p ${D3MOUTPUTDIR}/pipelines_ranked/$PIPELINE_ID.json \
    -r ${D3MINPUTDIR}/$DATASET/TRAIN/problem_TRAIN/problemDoc.json \
    -i ${D3MINPUTDIR}/$DATASET/TRAIN/dataset_TRAIN/datasetDoc.json \
    -t ${D3MINPUTDIR}/$DATASET/TEST/dataset_TEST/datasetDoc.json \
    -o ${D3MOUTPUTDIR}/predictions/$PIPELINE_ID.csv \
    -O ${D3MOUTPUTDIR}/pipeline_runs/$PIPELINE_ID.yaml \
    -n ${D3MOUTPUTDIR}/scoring_pipeline.yml \
    -a ${D3MINPUTDIR}/$DATASET/SCORE/dataset_TEST/datasetDoc.json \
    -c ${D3MOUTPUTDIR}/score/$PIPELINE_ID.score.csv

