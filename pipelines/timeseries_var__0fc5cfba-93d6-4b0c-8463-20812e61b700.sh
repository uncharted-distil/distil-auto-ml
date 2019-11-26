#!/bin/bash
python3 -m d3m runtime -v $D3MSTATICDIR -d $D3MINPUTDIR fit-score -r $D3MINPUTDIR/56_sunspots/56_sunspots_problem/problemDoc.json 	-i $D3MINPUTDIR/56_sunspots/TRAIN/dataset_TRAIN/datasetDoc.json -t $D3MINPUTDIR/56_sunspots/TEST/dataset_TEST/datasetDoc.json -a $D3MINPUTDIR/56_sunspots/SCORE/dataset_TEST/datasetDoc.json -p pipelines/timeseries_var__0fc5cfba-93d6-4b0c-8463-20812e61b700.json -O pipelines/timeseries_var__0fc5cfba-93d6-4b0c-8463-20812e61b700_run.yaml && 
gzip -f pipelines/timeseries_var__0fc5cfba-93d6-4b0c-8463-20812e61b700_run.yaml
