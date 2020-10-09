#!/bin/bash
python3 -m d3m runtime -v $D3MSTATICDIR -d $D3MINPUTDIR fit-score -r $D3MINPUTDIR/LL1_736_population_spawn_simpler_MIN_METADATA/LL1_736_population_spawn_simpler_MIN_METADATA_problem/problemDoc.json 	-i $D3MINPUTDIR/LL1_736_population_spawn_simpler_MIN_METADATA/TRAIN/dataset_TRAIN/datasetDoc.json -t $D3MINPUTDIR/LL1_736_population_spawn_simpler_MIN_METADATA/TEST/dataset_TEST/datasetDoc.json -a $D3MINPUTDIR/LL1_736_population_spawn_simpler_MIN_METADATA/SCORE/dataset_SCORE/datasetDoc.json -p pipelines/timeseries_var__f87551db-a099-4474-9baa-18bfe3a7b46f.json -O pipelines/timeseries_var__f87551db-a099-4474-9baa-18bfe3a7b46f_run.yaml && 
gzip -f pipelines/timeseries_var__f87551db-a099-4474-9baa-18bfe3a7b46f_run.yaml
