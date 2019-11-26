#!/bin/bash
python3 -m d3m runtime -v $D3MSTATICDIR -d $D3MINPUTDIR fit-score -r $D3MINPUTDIR/59_umls/59_umls_problem/problemDoc.json 	-i $D3MINPUTDIR/59_umls/TRAIN/dataset_TRAIN/datasetDoc.json -t $D3MINPUTDIR/59_umls/TEST/dataset_TEST/datasetDoc.json -a $D3MINPUTDIR/59_umls/SCORE/dataset_TEST/datasetDoc.json -p pipelines/link_prediction__082ba558-c2fb-4f75-bc36-19092f136a52.json -O pipelines/link_prediction__082ba558-c2fb-4f75-bc36-19092f136a52_run.yaml && 
gzip -f pipelines/link_prediction__082ba558-c2fb-4f75-bc36-19092f136a52_run.yaml
