#!/bin/bash
python3 -m d3m runtime -v $D3MSTATICDIR -d $D3MINPUTDIR fit-score -r $D3MINPUTDIR/30_personae_MIN_METADATA/30_personae_MIN_METADATA_problem/problemDoc.json 	-i $D3MINPUTDIR/30_personae_MIN_METADATA/TRAIN/dataset_TRAIN/datasetDoc.json -t $D3MINPUTDIR/30_personae_MIN_METADATA/TEST/dataset_TEST/datasetDoc.json -a $D3MINPUTDIR/30_personae_MIN_METADATA/SCORE/dataset_SCORE/datasetDoc.json -p pipelines/text__788860e9-ac6f-4a70-b84e-8982c969b1e1.json -O pipelines/text__788860e9-ac6f-4a70-b84e-8982c969b1e1_run.yaml && 
gzip -f pipelines/text__788860e9-ac6f-4a70-b84e-8982c969b1e1_run.yaml
