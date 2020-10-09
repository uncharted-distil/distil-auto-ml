#!/bin/bash
python3 -m d3m runtime -v $D3MSTATICDIR -d $D3MINPUTDIR fit-score -r $D3MINPUTDIR/LL1_50words_MIN_METADATA/LL1_50words_MIN_METADATA_problem/problemDoc.json 	-i $D3MINPUTDIR/LL1_50words_MIN_METADATA/TRAIN/dataset_TRAIN/datasetDoc.json -t $D3MINPUTDIR/LL1_50words_MIN_METADATA/TEST/dataset_TEST/datasetDoc.json -a $D3MINPUTDIR/LL1_50words_MIN_METADATA/SCORE/dataset_SCORE/datasetDoc.json -p pipelines/timeseries_kanine__a6002676-3c0d-4ae1-b909-ddf304b3cb34.json -O pipelines/timeseries_kanine__a6002676-3c0d-4ae1-b909-ddf304b3cb34_run.yaml && 
gzip -f pipelines/timeseries_kanine__a6002676-3c0d-4ae1-b909-ddf304b3cb34_run.yaml