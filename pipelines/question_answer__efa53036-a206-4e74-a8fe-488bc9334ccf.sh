#!/bin/bash
python3 -m d3m runtime -v $D3MSTATICDIR -d $D3MINPUTDIR fit-score -r $D3MINPUTDIR/32_wikiqa/32_wikiqa_problem/problemDoc.json 	-i $D3MINPUTDIR/32_wikiqa/TRAIN/dataset_TRAIN/datasetDoc.json -t $D3MINPUTDIR/32_wikiqa/TEST/dataset_TEST/datasetDoc.json -a $D3MINPUTDIR/32_wikiqa/SCORE/dataset_TEST/datasetDoc.json -p pipelines/question_answer__efa53036-a206-4e74-a8fe-488bc9334ccf.json -O pipelines/question_answer__efa53036-a206-4e74-a8fe-488bc9334ccf_run.yaml && 
gzip -f pipelines/question_answer__efa53036-a206-4e74-a8fe-488bc9334ccf_run.yaml
