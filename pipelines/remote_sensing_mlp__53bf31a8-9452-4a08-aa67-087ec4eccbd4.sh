#!/bin/bash
python3 -m d3m runtime -v $D3MSTATICDIR -d $D3MINPUTDIR fit-score -r $D3MINPUTDIR/locusts-nano/locusts-nano_problem/problemDoc.json 	-i $D3MINPUTDIR/locusts-nano/TRAIN/dataset_TRAIN/datasetDoc.json -t $D3MINPUTDIR/locusts-nano/TEST/dataset_TEST/datasetDoc.json -a $D3MINPUTDIR/locusts-nano/SCORE/dataset_SCORE/datasetDoc.json -p pipelines/remote_sensing_mlp__53bf31a8-9452-4a08-aa67-087ec4eccbd4.json -O pipelines/remote_sensing_mlp__53bf31a8-9452-4a08-aa67-087ec4eccbd4_run.yaml && 
gzip -f pipelines/remote_sensing_mlp__53bf31a8-9452-4a08-aa67-087ec4eccbd4_run.yaml
