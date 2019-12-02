#!/bin/bash
python3 -m d3m runtime -v $D3MSTATICDIR -d $D3MINPUTDIR fit-score -r $D3MINPUTDIR/49_facebook/49_facebook_problem/problemDoc.json 	-i $D3MINPUTDIR/49_facebook/TRAIN/dataset_TRAIN/datasetDoc.json -t $D3MINPUTDIR/49_facebook/TEST/dataset_TEST/datasetDoc.json -a $D3MINPUTDIR/49_facebook/SCORE/dataset_TEST/datasetDoc.json -p pipelines/graph_matching__4e24c9ee-3bdb-406b-9e34-8ae46228efa5.json -O pipelines/graph_matching__4e24c9ee-3bdb-406b-9e34-8ae46228efa5_run.yaml && 
gzip -f pipelines/graph_matching__4e24c9ee-3bdb-406b-9e34-8ae46228efa5_run.yaml
