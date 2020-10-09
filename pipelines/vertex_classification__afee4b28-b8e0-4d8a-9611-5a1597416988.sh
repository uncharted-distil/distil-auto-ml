#!/bin/bash
python3 -m d3m runtime -v $D3MSTATICDIR -d $D3MINPUTDIR fit-score -r $D3MINPUTDIR/LL1_VTXC_1369_synthetic_MIN_METADATA/LL1_VTXC_1369_synthetic_MIN_METADATA_problem/problemDoc.json 	-i $D3MINPUTDIR/LL1_VTXC_1369_synthetic_MIN_METADATA/TRAIN/dataset_TRAIN/datasetDoc.json -t $D3MINPUTDIR/LL1_VTXC_1369_synthetic_MIN_METADATA/TEST/dataset_TEST/datasetDoc.json -a $D3MINPUTDIR/LL1_VTXC_1369_synthetic_MIN_METADATA/SCORE/dataset_SCORE/datasetDoc.json -p pipelines/vertex_classification__afee4b28-b8e0-4d8a-9611-5a1597416988.json -O pipelines/vertex_classification__afee4b28-b8e0-4d8a-9611-5a1597416988_run.yaml && 
gzip -f pipelines/vertex_classification__afee4b28-b8e0-4d8a-9611-5a1597416988_run.yaml