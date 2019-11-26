#!/bin/bash
python3 -m d3m runtime -v $D3MSTATICDIR -d $D3MINPUTDIR fit-score -r $D3MINPUTDIR/31_urbansound/31_urbansound_problem/problemDoc.json 	-i $D3MINPUTDIR/31_urbansound/TRAIN/dataset_TRAIN/datasetDoc.json -t $D3MINPUTDIR/31_urbansound/TEST/dataset_TEST/datasetDoc.json -a $D3MINPUTDIR/31_urbansound/SCORE/dataset_TEST/datasetDoc.json -p pipelines/audio__650d7784-14c0-42e5-977e-59a3525d4c08.json -O pipelines/audio__650d7784-14c0-42e5-977e-59a3525d4c08_run.yaml && 
gzip -f pipelines/audio__650d7784-14c0-42e5-977e-59a3525d4c08_run.yaml
