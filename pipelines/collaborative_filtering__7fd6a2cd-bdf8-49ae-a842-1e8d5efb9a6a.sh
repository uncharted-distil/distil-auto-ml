#!/bin/bash
python3 -m d3m runtime -v $D3MSTATICDIR -d $D3MINPUTDIR fit-score -r $D3MINPUTDIR/60_jester/60_jester_problem/problemDoc.json 	-i $D3MINPUTDIR/60_jester/TRAIN/dataset_TRAIN/datasetDoc.json -t $D3MINPUTDIR/60_jester/TEST/dataset_TEST/datasetDoc.json -a $D3MINPUTDIR/60_jester/SCORE/dataset_TEST/datasetDoc.json -p pipelines/collaborative_filtering__7fd6a2cd-bdf8-49ae-a842-1e8d5efb9a6a.json -O pipelines/collaborative_filtering__7fd6a2cd-bdf8-49ae-a842-1e8d5efb9a6a_run.yaml && 
gzip -f pipelines/collaborative_filtering__7fd6a2cd-bdf8-49ae-a842-1e8d5efb9a6a_run.yaml
