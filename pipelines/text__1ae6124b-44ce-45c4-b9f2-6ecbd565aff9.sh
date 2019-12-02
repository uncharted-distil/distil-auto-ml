#!/bin/bash
python3 -m d3m runtime -v $D3MSTATICDIR -d $D3MINPUTDIR fit-score -r $D3MINPUTDIR/30_personae/30_personae_problem/problemDoc.json 	-i $D3MINPUTDIR/30_personae/TRAIN/dataset_TRAIN/datasetDoc.json -t $D3MINPUTDIR/30_personae/TEST/dataset_TEST/datasetDoc.json -a $D3MINPUTDIR/30_personae/SCORE/dataset_TEST/datasetDoc.json -p pipelines/text__1ae6124b-44ce-45c4-b9f2-6ecbd565aff9.json -O pipelines/text__1ae6124b-44ce-45c4-b9f2-6ecbd565aff9_run.yaml && 
gzip -f pipelines/text__1ae6124b-44ce-45c4-b9f2-6ecbd565aff9_run.yaml
