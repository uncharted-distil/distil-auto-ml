#!/bin/bash
python3 -m d3m runtime -v $D3MSTATICDIR -d $D3MINPUTDIR fit-score -r $D3MINPUTDIR/30_personae_MIN_METADATA/30_personae_MIN_METADATA_problem/problemDoc.json 	-i $D3MINPUTDIR/30_personae_MIN_METADATA/TRAIN/dataset_TRAIN/datasetDoc.json -t $D3MINPUTDIR/30_personae_MIN_METADATA/TEST/dataset_TEST/datasetDoc.json -a $D3MINPUTDIR/30_personae_MIN_METADATA/SCORE/dataset_SCORE/datasetDoc.json -p pipelines/text__642982fc-7e43-4426-ba3d-3d18132929e1.json -O pipelines/text__642982fc-7e43-4426-ba3d-3d18132929e1_run.yaml && 
gzip -f pipelines/text__642982fc-7e43-4426-ba3d-3d18132929e1_run.yaml
