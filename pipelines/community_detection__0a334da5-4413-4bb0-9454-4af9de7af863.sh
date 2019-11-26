#!/bin/bash
python3 -m d3m runtime -v $D3MSTATICDIR -d $D3MINPUTDIR fit-score -r $D3MINPUTDIR/6_70_com_amazon/6_70_com_amazon_problem/problemDoc.json 	-i $D3MINPUTDIR/6_70_com_amazon/TRAIN/dataset_TRAIN/datasetDoc.json -t $D3MINPUTDIR/6_70_com_amazon/TEST/dataset_TEST/datasetDoc.json -a $D3MINPUTDIR/6_70_com_amazon/SCORE/dataset_TEST/datasetDoc.json -p pipelines/community_detection__0a334da5-4413-4bb0-9454-4af9de7af863.json -O pipelines/community_detection__0a334da5-4413-4bb0-9454-4af9de7af863_run.yaml && 
gzip -f pipelines/community_detection__0a334da5-4413-4bb0-9454-4af9de7af863_run.yaml
