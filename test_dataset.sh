export DATASET=$1
export NUMROWS=${2:-1000}
#  some datasets are big, it can be nice to test a subset.
#mv ./seed_datasets_current/$DATASET/${DATASET}_dataset/tables/learningData.csv ./seed_datasets_current/$DATASET/${DATASET}_dataset/tables/learningData.csv2
#head -n $NUMROWS ./seed_datasets_current/$DATASET/${DATASET}_dataset/tables/learningData.csv2 > ./seed_datasets_current/$DATASET/${DATASET}_dataset/tables/learningData.csv
python -m dummy_ta3.dummy_ta3 -p ./seed_datasets_current/$DATASET/TRAIN/problem_TRAIN/problemDoc.json -d ./seed_datasets_current -e 0.0.0.0 -t 45042

echo "Ran search successfully!"
#mv seed_datasets_current/$DATASET/${DATASET}_dataset/tables/learningData.csv2 seed_datase#
eval $(cat pipeline_id.txt | sed 's/^/export /')

export D3MINPUTDIR=seed_datasets_current
export D3MOUTPUTDIR=output
export D3MSTATICDIR="/static"

[ -z "$D3MINPUTDIR" ] && echo "Error - D3MINPUTDIR unset"
[ -z "$D3MOUTPUTDIR" ] && echo "Error - D3MOUTPUTDIR unset"
[ -z "$D3MSTATICDIR" ] && echo "Error - D3MSTATICDIR unset"
[ -z "$DATASET" ] && echo "Error - DATASET unset"
[ -z "$PIPELINE_ID" ] && echo "Error - PIPELINE_ID unset"

mkdir -p ${D3MOUTPUTDIR}/logs && \
mkdir -p ${D3MOUTPUTDIR}/predictions && \
mkdir -p ${D3MOUTPUTDIR}/pipeline_runs && \
mkdir -p ${D3MOUTPUTDIR}/score

cp ./test_scripts/scoring_pipeline.yml ${D3MOUTPUTDIR}

#mv ./seed_datasets_current/$DATASET/TEST/dataset_TEST/tables/learningData.csv ./seed_datasets_current/$DATASET/TEST/dataset_TEST/tables/learningData.csv2
#head -n $NUMROWS ./seed_datasets_current/$DATASET/TEST/dataset_TEST/tables/learningData.csv2 > ./seed_datasets_current/$DATASET/TEST/dataset_TEST/tables/learningData.csv
####shuf -n $NUMROWS ./seed_datasets_current/$DATASET/TEST/dataset_TEST/tables/learningData.csv2 >> ./seed_datasets_current/$DATASET/TEST/dataset_TEST/tables/learningData.csv
####
#mv ./seed_datasets_current/$DATASET/TRAIN/dataset_TRAIN/tables/learningData.csv ./seed_datasets_current/$DATASET/TRAIN/dataset_TRAIN/tables/learningData.csv2
#head -n $NUMROWS ./seed_datasets_current/$DATASET/TRAIN/dataset_TRAIN/tables/learningData.csv2 > ./seed_datasets_current/$DATASET/TRAIN/dataset_TRAIN/tables/learningData.csv
####shuf -n $NUMROWS ./seed_datasets_current/$DATASET/TRAIN/dataset_TRAIN/tables/learningData.csv2 >> ./seed_datasets_current/$DATASET/TRAIN/dataset_TRAIN/tables/learningData.csv
###


FILE=${D3MINPUTDIR}/$DATASET/SCORE/dataset_SCORE/datasetDoc.json
if test -f "$FILE"; then
  python -m d3m runtime \
      --volumes "${D3MSTATICDIR}" \
      -d ${DATASET} \
      fit-score \
      -p ${D3MOUTPUTDIR}/pipelines_ranked/$PIPELINE_ID.json \
      -r ${D3MINPUTDIR}/$DATASET/${DATASET}_problem/problemDoc.json \
      -i ${D3MINPUTDIR}/$DATASET/TRAIN/dataset_TRAIN/datasetDoc.json \
      -t ${D3MINPUTDIR}/$DATASET/TEST/dataset_TEST/datasetDoc.json \
      -a ${D3MINPUTDIR}/$DATASET/SCORE/dataset_SCORE/datasetDoc.json
else
  python -m d3m runtime \
      --volumes "${D3MSTATICDIR}" \
      -d ${DATASET} \
      fit-score \
      -p ${D3MOUTPUTDIR}/pipelines_ranked/$PIPELINE_ID.json \
      -r ${D3MINPUTDIR}/$DATASET/${DATASET}_problem/problemDoc.json \
      -i ${D3MINPUTDIR}/$DATASET/TRAIN/dataset_TRAIN/datasetDoc.json \
      -t ${D3MINPUTDIR}/$DATASET/TEST/dataset_TEST/datasetDoc.json \
      -a ${D3MINPUTDIR}/$DATASET/SCORE/dataset_TEST/datasetDoc.json
fi

echo "Ran score successfully!"
#
##when done
#mv ./seed_datasets_current/$DATASET/TEST/dataset_TEST/tables/learningData.csv2 ./seed_datasets_current/$DATASET/TEST/dataset_TEST/tables/learningData.csv
#mv ./seed_datasets_current/$DATASET/TRAIN/dataset_TRAIN/tables/learningData.csv2 ./seed_datasets_current/$DATASET/TRAIN/dataset_TRAIN/tables/learningData.csv
####
##
#
#ts_current/$DATASET/${DATASET}_dataset/tables/learningData.csv
