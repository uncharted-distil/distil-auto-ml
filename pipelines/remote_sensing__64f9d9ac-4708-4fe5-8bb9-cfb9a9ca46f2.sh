#!/bin/bash
python3 -m d3m runtime -v $D3MSTATICDIR -d $D3MINPUTDIR fit-score -r $D3MINPUTDIR/big-earth-sample/big-earth-sample_problem/problemDoc.json 	-i $D3MINPUTDIR/big-earth-sample/TRAIN/dataset_TRAIN/datasetDoc.json -t $D3MINPUTDIR/big-earth-sample/TEST/dataset_TEST/datasetDoc.json -a $D3MINPUTDIR/big-earth-sample/SCORE/dataset_SCORE/datasetDoc.json -p pipelines/remote_sensing__64f9d9ac-4708-4fe5-8bb9-cfb9a9ca46f2.json -O pipelines/remote_sensing__64f9d9ac-4708-4fe5-8bb9-cfb9a9ca46f2_run.yaml && 
gzip -f pipelines/remote_sensing__64f9d9ac-4708-4fe5-8bb9-cfb9a9ca46f2_run.yaml
