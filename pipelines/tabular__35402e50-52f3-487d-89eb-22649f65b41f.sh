#!/bin/bash
python3 -m d3m runtime -v $D3MSTATICDIR -d $D3MINPUTDIR fit-score -r $D3MINPUTDIR/LL0_acled_reduced/LL0_acled_reduced_problem/problemDoc.json 	-i $D3MINPUTDIR/LL0_acled_reduced/TRAIN/dataset_TRAIN/datasetDoc.json -t $D3MINPUTDIR/LL0_acled_reduced/TEST/dataset_TEST/datasetDoc.json -a $D3MINPUTDIR/LL0_acled_reduced/SCORE/dataset_TEST/datasetDoc.json -p pipelines/tabular__35402e50-52f3-487d-89eb-22649f65b41f.json -O pipelines/tabular__35402e50-52f3-487d-89eb-22649f65b41f_run.yaml && 
gzip -f pipelines/tabular__35402e50-52f3-487d-89eb-22649f65b41f_run.yaml
