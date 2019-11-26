#!/bin/bash
python3 -m d3m runtime -v $D3MSTATICDIR -d $D3MINPUTDIR fit-score -r $D3MINPUTDIR/22_handgeometry/22_handgeometry_problem/problemDoc.json 	-i $D3MINPUTDIR/22_handgeometry/TRAIN/dataset_TRAIN/datasetDoc.json -t $D3MINPUTDIR/22_handgeometry/TEST/dataset_TEST/datasetDoc.json -a $D3MINPUTDIR/22_handgeometry/SCORE/dataset_TEST/datasetDoc.json -p pipelines/image__e802f2ed-c035-40b7-b2b8-7426ac8de3d5.json -O pipelines/image__e802f2ed-c035-40b7-b2b8-7426ac8de3d5_run.yaml && 
gzip -f pipelines/image__e802f2ed-c035-40b7-b2b8-7426ac8de3d5_run.yaml
