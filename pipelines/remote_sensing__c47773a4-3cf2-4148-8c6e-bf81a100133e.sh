#!/bin/bash
python3 -m d3m runtime -v $D3MSTATICDIR -d $D3MINPUTDIR fit-score -r $D3MINPUTDIR/big-earth-sample/big-earth-sample_problem/problemDoc.json 	-i $D3MINPUTDIR/big-earth-sample/TRAIN/dataset_TRAIN/datasetDoc.json -t $D3MINPUTDIR/big-earth-sample/TEST/dataset_TEST/datasetDoc.json -a $D3MINPUTDIR/big-earth-sample/SCORE/dataset_SCORE/datasetDoc.json -p pipelines/remote_sensing__c47773a4-3cf2-4148-8c6e-bf81a100133e.json -O pipelines/remote_sensing__c47773a4-3cf2-4148-8c6e-bf81a100133e_run.yaml && 
gzip -f pipelines/remote_sensing__c47773a4-3cf2-4148-8c6e-bf81a100133e_run.yaml
