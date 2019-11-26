#!/bin/bash
python3 -m d3m runtime -v $D3MSTATICDIR -d $D3MINPUTDIR fit-score -r $D3MINPUTDIR/LL1_50words/LL1_50words_problem/problemDoc.json 	-i $D3MINPUTDIR/LL1_50words/TRAIN/dataset_TRAIN/datasetDoc.json -t $D3MINPUTDIR/LL1_50words/TEST/dataset_TEST/datasetDoc.json -a $D3MINPUTDIR/LL1_50words/SCORE/dataset_TEST/datasetDoc.json -p pipelines/timeseries_kanine__96ee922d-dbdd-4f7c-a33e-8d98a05ba007.json -O pipelines/timeseries_kanine__96ee922d-dbdd-4f7c-a33e-8d98a05ba007_run.yaml && 
gzip -f pipelines/timeseries_kanine__96ee922d-dbdd-4f7c-a33e-8d98a05ba007_run.yaml
